from agent_graph.lang_graph_trial.state_graph import graph
from root import data_output_path

# Save the graph as a PNG file
graph_image_path = data_output_path() + "graph_output.png"
graph.get_graph().draw_mermaid_png(output_file_path=str(graph_image_path))

# Display the image if running in an environment that supports GUI (like PyCharm)
# import webbrowser
# webbrowser.open(graph_image_path.as_uri())
