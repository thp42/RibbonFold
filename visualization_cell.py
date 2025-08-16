# Jupyter notebook cell for RibbonFold confidence visualization
# Copy this cell to your RibbonFold_v0_2.ipynb notebook

cell_content = '''
#@title 12. Generate and Display Confidence Plots {run: "auto"}
import os
import sys
import json
import glob
from IPython.display import display, HTML
import base64
from html import escape

# Add the plotting module to the path
sys.path.append('/content/ribbonfold')

# Import the plotting functions
exec(open('/content/ribbonfold/plot_confidence.py').read())

# Configuration
RESULTS_DIR = "/content/ribbonfold/results"  # Update this path if needed

# Function to convert image to data URL for HTML display
def image_to_data_url(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        img = f.read()
    return prefix + base64.b64encode(img).decode('utf-8')

# Find the most recent results directory (if multiple exist)
if os.path.exists(RESULTS_DIR):
    result_subdirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    if result_subdirs:
        # Use the most recent directory
        result_subdirs.sort()
        actual_results_dir = os.path.join(RESULTS_DIR, result_subdirs[-1])
    else:
        actual_results_dir = RESULTS_DIR
else:
    actual_results_dir = RESULTS_DIR

print(f"Looking for results in: {actual_results_dir}")

# Check if results directory exists
if not os.path.exists(actual_results_dir):
    print(f"Results directory not found: {actual_results_dir}")
    print("Make sure inference has completed successfully.")
else:
    # Create confidence plots
    try:
        plot_files = create_confidence_plots(actual_results_dir)
        
        if plot_files:
            # Get job name from results directory
            jobname = os.path.basename(actual_results_dir.rstrip('/'))
            
            # Prepare image data URLs
            pae = ""
            plddt = ""
            coverage = ""
            
            if 'pae' in plot_files and os.path.isfile(plot_files['pae']):
                pae = image_to_data_url(plot_files['pae'])
            
            if 'plddt' in plot_files and os.path.isfile(plot_files['plddt']):
                plddt = image_to_data_url(plot_files['plddt'])
                
            if 'coverage' in plot_files and os.path.isfile(plot_files['coverage']):
                coverage = image_to_data_url(plot_files['coverage'])
            
            # Display the plots using HTML
            display(HTML(f"""
            <style>
              img {{
                float:left;
                margin: 10px;
              }}
              .full {{
                max-width:90%;
                clear: both;
              }}
              .half {{
                max-width:45%;
              }}
              @media (max-width:640px) {{
                .half {{
                  max-width:100%;
                }}
              }}
              .plot-container {{
                margin-bottom: 20px;
                overflow: hidden;
              }}
            </style>
            <div style="max-width:95%; padding:2em;">
              <h1>Confidence Plots for {escape(jobname)}</h1>
              <div class="plot-container">
                { '<!--' if pae == '' else '' }<img src="{pae}" class="full" alt="PAE Plot" />{ '-->' if pae == '' else '' }
              </div>
              <div class="plot-container">
                <img src="{plddt}" class="half" alt="pLDDT Plot" />
                <img src="{coverage}" class="half" alt="Coverage Plot" />
              </div>
              <div style="clear: both; margin-top: 20px;">
                <p><strong>Plot descriptions:</strong></p>
                <ul>
                  <li><strong>pLDDT Plot:</strong> Per-residue confidence scores (0-100). Higher scores indicate more reliable predictions.</li>
                  { '<li><strong>PAE Plot:</strong> Predicted Aligned Error between residue pairs. Lower values (darker) indicate more confident relative positions.</li>' if pae != '' else '' }
                  <li><strong>Coverage Plot:</strong> MSA coverage information (placeholder - shows estimated coverage pattern).</li>
                </ul>
              </div>
            </div>
            """))
            
            print("\\nPlot files created:")
            for plot_type, plot_file in plot_files.items():
                print(f"  {plot_type}: {plot_file}")
                
        else:
            print("No confidence data found or plots could not be created.")
            print("Make sure the inference completed successfully and confidence JSON files exist.")
            
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
'''

print("=== Jupyter Notebook Cell for Confidence Visualization ===")
print("Copy the following cell to your RibbonFold_v0_2.ipynb notebook:")
print("=" * 60)
print(cell_content)
