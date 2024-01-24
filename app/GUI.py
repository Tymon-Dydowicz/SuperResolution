import gradio as gr

def clear():
    return None, None

def generateGUI(availableModels, processImageFunction, visualizeTraining):
    with gr.Blocks() as GUI:
        modelSelection = gr.Dropdown(availableModels, label="Model")
        inputImage = gr.Image(label="Test Image", sources=['upload'])
        outputImage = gr.Image(height=256, width=256, label="Output Image")
        processButton = gr.Button("Process")
        clearButton = gr.Button("Clear")
        clearButton.click(fn=clear, inputs=None, outputs=[inputImage, outputImage], api_name="clear")
        processButton.click(fn=processImageFunction, inputs=[inputImage, modelSelection], outputs=outputImage, api_name="process")
        
        # visualizeModelButton = gr.Button("Visualize Model")
        visualizeMetricsButton = gr.Button("Show training metrics")
        # modelStructure = gr.Textbox(label="Model Structure")
        metricsVisualization = gr.Plot(label="Metrics Visualization")

        # visualizeModelButton.click(fn=visualize_model, inputs=modelSelection, outputs=modelStructure, api_name="visualize_model")
        visualizeMetricsButton.click(fn=visualizeTraining, inputs=modelSelection, outputs=metricsVisualization, api_name="visualize_metrics")
    
    return GUI