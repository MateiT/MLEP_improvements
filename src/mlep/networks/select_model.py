def select_network(opt):
    # network = opt['network']['name']
    network = 'network_basic'
    if network == 'network_basic':
        from mlep.networks.network_basic import ResTransformer
    # elif network == 'network_pooling':
    #     from mlep.networks.network_pooling import ResTransformer
    # elif network == 'network_maxpooling':
    #     from mlep.networks.network_maxpooling import ResTransformer
    # elif network == 'network_vit':
    #     from mlep.networks.network_vit import ResTransformer
    # elif network == 'network_noT':
    #     from mlep.networks.network_noT import ResTransformer
    # elif network == 'network_gate':
    #     from mlep.networks.network_gate import ResTransformer
    # elif network == 'network_gate2':
    #     from mlep.networks.network_gate2 import ResTransformer
    # elif network == 'network_conv':
    #     from mlep.networks.network_conv import ResTransformer
    # elif network == 'network_unet':
    #     from mlep.networks.network_unet import ResTransformer
    return ResTransformer()

