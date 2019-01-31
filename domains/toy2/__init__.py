"""
Toy2 dataset is composed of an XML tree containing two randomly generated addresses.
The first address is labelled billTo while the other is shipTo.
The XML schema can be seen inside schema.xsd file.

The output transformation is swapping of billTo and shipTo addresses.
"""
appConfigDefaults = {
    # AppConfig defaults
}

modelArgsDefaults = {
    "attrib_value_vec_len": 32,
    "node_info_propagator_stack_depth": 3,
    "propagated_info_len": 64,
    "output_decoder_stack_depth": 1,
    "output_decoder_state_width": 100,
}
