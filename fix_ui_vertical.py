import re

def rewrite_generate_plot():
    with open('app.py', 'r') as f:
        content = f.read()

    new_func = '''
def generate_plot_and_analysis(df, progress, stimulus_type="Text", stimulus_desc=""):
    """Generate vertical brain plot AND AI analysis, then persist the run."""
    import matplotlib.pyplot as plt
    from tribev2.plotting.utils import robust_normalize
    
    progress((0.4, 1.0), desc="Extracting Deep Multimodal AI Features (Heaviest Step)...")
    m = get_model()
    preds, segments = m.predict(events=df, gradio_progress=progress)
    
    progress((0.75, 1.0), desc="Rendering 3D Brain Mesh (Vertical Layout)...")
    n_to_plot = min(len(preds), 4)
    sliced_preds = preds[:n_to_plot]
    views_seq = ["left", "right", "dorsal", "anterior"]
    
    # Normalize data globally for consistent colorbar
    norm_preds = [robust_normalize(p, percentile=95) for p in sliced_preds]
    
    fig = plt.figure(figsize=(10, 3.5 * n_to_plot))
    fig.patch.set_facecolor('#09090b')
    
    sm = None
    for i in range(n_to_plot):
        view_name = views_seq[i]
        view_title, view_desc = VIEW_LABELS.get(view_name, ("Brain View", "Neural activation"))
        raw_act = float(np.mean(np.abs(sliced_preds[i])))
        
        # Brain plot on the left
        ax_brain = fig.add_subplot(n_to_plot, 2, 2*i + 1, projection="3d")
        ax_brain.set_facecolor('#09090b')
        
        sm = plotter.plot_surf(
            norm_preds[i],
            views=view_name,
            axes=[ax_brain], # Note: plot_surf handles lists of axes inside cortical.py if we pass a list? Or just the ax. 
            # wait, plot_surf in cortical.py: if views is a string, and axes is a single ax? No, if axes is not None, it uses get_axarr_and_views.
            # let's just pass axes=ax_brain. Wait, cortical.py expects axes to be a list if views is a list.
            # I will pass [ax_brain]
            colorbar=False,
            cmap="hot"
        )
        
        # Text on the right
        ax_text = fig.add_subplot(n_to_plot, 2, 2*i + 2)
        ax_text.axis("off")
        ax_text.set_facecolor('#09090b')
        
        label_text = f"{view_title}\n\n{view_desc}\n\nActivation: {raw_act * 100:.1f}%"
        ax_text.text(0.1, 0.5, label_text, fontsize=14, color="#e4e4e7", va="center", ha="left")

    # Add a global colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.25, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Relative Activation Intensity", color="#a1a1aa", fontsize=10)
    cbar.ax.tick_params(colors="#a1a1aa", labelsize=9)
    # remove ticks
    cbar.set_ticks([])
    
    fig.subplots_adjust(top=0.95, bottom=0.12, hspace=0.1)

    progress((0.9, 1.0), desc="Analyzing Brain Activation Patterns...")
    interpretation, region_data = analyze_brain_regions(preds, stimulus_desc)
    
    progress((0.95, 1.0), desc="Saving run to history...")
    pdf_path = None
    try:
        _, pdf_path = save_run(stimulus_type, stimulus_desc, fig, interpretation, region_data)
    except Exception as e:
        print(f"[Run History] Failed to save: {e}")
    
    progress((1.0, 1.0), desc="Complete")
    return fig, interpretation, gr.update(value=pdf_path, visible=bool(pdf_path))
'''
    import re
    
    pattern = re.compile(r'def generate_plot_and_analysis\(.*?return fig, interpretation, gr\.update\(value=pdf_path, visible=bool\(pdf_path\)\)', re.DOTALL)
    
    if pattern.search(content):
        content = pattern.sub(new_func.strip(), content)
        with open('app.py', 'w') as f:
            f.write(content)
        print("Replaced generate_plot_and_analysis")
    else:
        print("Could not find function to replace!")

rewrite_generate_plot()
