"""
Toy1 dataset is composed of simple single node XML trees with a randomly generated text
property. The output for each tree is the same tree with the text property reversed.
For example

<toyrev>kldasd</toyrev> becomes <toyrev>dsadlk</toyrev>
"""

appConfigDefaults = {
    # AppConfig defaults
    "checkpoint_every": 10,
}
modelArgsDefaults = {
    "attrib_value_vec_len": 32,
    "node_info_propagator_stack_depth": 3,
    "propagated_info_len": 128,
    "output_decoder_stack_depth": 1,
    "output_decoder_state_width": 128,
}
