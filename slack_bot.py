import os
import sys
import logging
import threading
import requests
from dotenv import load_dotenv
from pathlib import Path
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tribe-slack")

# ── Environment ──────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "").strip()
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "").strip()
MODAL_WEBHOOK_URL = os.environ.get("MODAL_WEBHOOK_URL", "").strip()

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    logger.error("Missing Slack tokens in .env. Fix this before running.")
    sys.exit(1)

# ── Video file extensions we process ─────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── Slack App ────────────────────────────────────────────────────────
slack_app = App(token=SLACK_BOT_TOKEN)

# Track files we're currently processing to avoid duplicates
_processing_lock = threading.Lock()
_processing_files: set[str] = set()

# ── Slack Event Handlers ─────────────────────────────────────────────
@slack_app.event("message")
def handle_message(event, client):
    """Process video files dropped into any channel the bot is in."""
    logger.info(f"Received message event: subtype={event.get('subtype')} user={event.get('user')}")
    
    # Ignore bot messages (including our own uploads) - explicit check against bot_message
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        logger.info("Ignoring bot message.")
        return

    files = event.get("files", [])
    if not files:
        logger.info("No files in message, ignoring.")
        return

    channel = event["channel"]
    thread_ts = event.get("ts")  # We'll reply in-thread

    for file_info in files:
        filename = file_info.get("name", "unknown")
        ext = Path(filename).suffix.lower()

        logger.info(f"Found file: {filename} (ext: {ext})")
        if ext not in VIDEO_EXTENSIONS:
            logger.info(f"Skipping non-video file: {filename}")
            continue

        file_id = file_info["id"]

        # Deduplicate
        with _processing_lock:
            if file_id in _processing_files:
                continue
            _processing_files.add(file_id)

        # Process in a background thread
        t = threading.Thread(
            target=_forward_video_to_modal,
            args=(client, channel, thread_ts, file_info, filename, file_id),
            daemon=True,
        )
        t.start()


def _forward_video_to_modal(client, channel, thread_ts, file_info, filename, file_id):
    """Send video info to our Serverless GPU on Modal."""
    try:
        if not MODAL_WEBHOOK_URL:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="⚠️ *Missing AI Backend URL.* The bot listener is running, but `MODAL_WEBHOOK_URL` is missing."
            )
            return

        # ── Get download URL ──
        download_url = file_info.get("url_private_download")
        if not download_url:
            try:
                result = client.files_info(file=file_id)
                download_url = result["file"].get("url_private_download")
            except Exception as e:
                logger.error(f"Error fetching file info: {e}")

        if not download_url:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="⚠️ *Silent Failure Caught*\nSlack hasn't generated a download URL for this file yet. It might still be natively uploading/processing on Slack's end."
            )
            return

        # ── Post "processing" status ──
        processing_msg = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=(f"🧠 *TRIBE v2 Brain Encoding* — `{filename}`\n"
                  f"⏳ _Forwarding to Modal AI Backend..._")
        )
        proc_ts = processing_msg["ts"]

        # Add hourglass reaction
        try:
            client.reactions_add(channel=channel, timestamp=thread_ts, name="hourglass_flowing_sand")
        except Exception:
            pass

        # ── Payload for Modal ──
        payload = {
            "channel": channel,
            "thread_ts": thread_ts,
            "proc_ts": proc_ts,
            "filename": filename,
            "download_url": download_url,
        }

        logger.info(f"Forwarding {filename} to Modal Serverless GPU...")
        
        # Fire and forget POST request (Modal will return 200 via web_endpoint immediately)
        # We specify a short timeout to prevent the thread hanging if Modal struggles
        response = requests.post(MODAL_WEBHOOK_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully triggered Modal for {filename}.")
        else:
            raise Exception(f"Modal rejected payload: HTTP {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Failed to forward to Modal: {e}", exc_info=True)
        try:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(f"❌ *AI Backend Error*\n"
                      f"Failed to trigger Modal Serverless GPU for `{filename}`.")
            )
            client.reactions_remove(channel=channel, timestamp=thread_ts, name="hourglass_flowing_sand")
            client.reactions_add(channel=channel, timestamp=thread_ts, name="x")
        except Exception:
            pass
    finally:
        with _processing_lock:
            _processing_files.discard(file_id)

@slack_app.event("file_shared")
def handle_file_shared(event):
    pass

# ── Welcome Banner when bot joins a channel ──────────────────────────
WELCOME_BANNER = Path(__file__).parent / "assets" / "welcome_banner.png"

@slack_app.event("member_joined_channel")
def handle_bot_joined(event, client):
    """Post a welcome banner when the bot is added to a channel."""
    user_id = event.get("user")
    channel = event.get("channel")

    # Only fire when it's the bot itself joining, not other users
    try:
        auth = client.auth_test()
        bot_id = auth["user_id"]
    except Exception:
        return

    if user_id != bot_id:
        return

    logger.info(f"Bot was added to channel {channel}. Posting welcome banner.")

    welcome_text = (
        "🧠 *NeuralAI by Multitude Media*\n\n"
        "Hey team! I'm your *Neural Engagement Scorecard* bot.\n\n"
        "*How it works:*\n"
        "1️⃣  Drop any `.mp4` video into this channel\n"
        "2️⃣  I'll fire it through a serverless GPU running Meta's V-JEPA2 neural encoder\n"
        "3️⃣  You'll get a professional PDF scorecard with second-by-second brain activation analytics\n\n"
        "_Powered by Facebook's TRIBE v2 multimodal brain mapping model._"
    )

    try:
        if WELCOME_BANNER.exists():
            client.files_upload_v2(
                channel=channel,
                file=str(WELCOME_BANNER),
                filename="NeuralAI_Welcome.png",
                title="NeuralAI by Multitude Media",
                initial_comment=welcome_text,
            )
        else:
            client.chat_postMessage(channel=channel, text=welcome_text)
    except Exception as e:
        logger.error(f"Failed to post welcome banner: {e}")

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🚀  TRIBE v2 Socket Listener (Render)          ║")
    print("║                                                          ║")
    print("║  Listening for videos natively on your Mac/Render Server ║")
    print("║  Forwarding all ML extraction to Serverless GPUs...      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if MODAL_WEBHOOK_URL:
        logger.info(f"Modal Backend: {MODAL_WEBHOOK_URL}")
    else:
        logger.warning(f"WAITING: MODAL_WEBHOOK_URL is not set yet in .env! Ensure you deploy Modal first.")

    logger.info("Connecting to Slack via Socket Mode...")
    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()
