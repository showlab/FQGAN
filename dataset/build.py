from dataset.imagenet import build_imagenet, build_imagenet_code, build_imagenet_triple_code, build_imagenet_dual_code


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'imagenet_dual_code':
        return build_imagenet_dual_code(args, **kwargs)
    if args.dataset == 'imagenet_triple_code':
        return build_imagenet_triple_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')
