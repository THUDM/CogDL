from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    remask_rate = args.remask_rate
    mask_method = args.mask_method
    drop_edge_rate = args.drop_edge_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l

    num_features = args.num_features
    num_dec_layers = args.num_dec_layers
    num_remasking = args.num_remasking
    lam = args.lam
    delayed_ema_epoch = args.delayed_ema_epoch
    replace_rate = args.replace_rate
    remask_method = args.remask_method
    momentum = args.momentum

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        num_remasking=num_remasking,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        remask_rate=remask_rate,
        mask_method=mask_method,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        alpha_l=alpha_l,
        lam=lam,
        delayed_ema_epoch=delayed_ema_epoch,
        replace_rate=replace_rate,
        remask_method=remask_method,
        momentum=momentum
    )
    return model
