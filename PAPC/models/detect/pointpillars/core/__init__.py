import numpy as np
from functools import partial
import pickle 

import paddle

from core.box_coders import (GroundBox3dCoderPaddle, BevBoxCoderPaddle)
from core.voxel_generator import VoxelGenerator
from core.target_assigner import TargetAssigner
from core.similarity_calculator import (RotateIouSimilarity,NearestIouSimilarity,DistanceSimilarity)
from core.anchor_generator import (AnchorGeneratorStride, AnchorGeneratorRange)
from core import losses 

from data.dataset import KittiDataset, DatasetWrapper
from data.preprocess import prep_pointcloud
from libs.preprocess import DBFilterByMinNumPoint, DBFilterByDifficulty, DataBasePreprocessor
from libs.ops.sample_ops import DataBaseSamplerV2
from libs.tools import learning_schedules


def build_anchor_generator(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """

    if  'anchor_generator_stride' in anchor_config:
        config = anchor_config.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    elif 'anchor_generator_range' in anchor_config:
        config = anchor_config.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    else:
        raise ValueError(" unknown anchor generator type")


def build_similarity_calculator(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """

    if 'rotate_iou_similarity' in similarity_config:
        return RotateIouSimilarity()
    elif 'nearest_iou_similarity' in similarity_config:
        return NearestIouSimilarity()
    elif 'distance_similarity' in similarity_config:
        cfg = similarity_config.distance_similarity
        return DistanceSimilarity(distance_norm=cfg.distance_norm,
                                  with_rotation=cfg.with_rotation,
                                  rotation_alpha=cfg.rotation_alpha)
    else:
        raise ValueError("unknown similarity type")


def build_voxel_generator(voxel_generator_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """

    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_generator_config.VOXEL_SIZE),
        point_cloud_range=list(voxel_generator_config.POINT_CLOUD_RANGE),
        max_num_points=voxel_generator_config.MAX_NUMBER_OF_POINTS_PER_VOXEL,
        max_voxels=voxel_generator_config.MAX_VOXELS)
    return voxel_generator


def build_box_coder(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """

    if box_coder_config.BOX_CODER_TYPE == 'ground_box3d_coder':
 
        return GroundBox3dCoderPaddle(box_coder_config.LINEAR_DIM, box_coder_config.ENCODE_ANGLE_VECTOR)
    elif box_coder_config.BOX_CODER_TYPE == 'bev_box_coder':
        return BevBoxCoderPaddle(box_coder_config.LINEAR_DIM, box_coder_config.ENCODE_ANGLE_VECTOR,
                                box_coder_config.Z_FIXED, box_coder_config.H_FIXED)
    else:
        raise ValueError("unknown box_coder type")
    

def build_target_assigner(target_assigner_config, bv_range, box_coder):

    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """

    anchor_cfg = target_assigner_config.ANCHOR_GENERATORS
    anchor_generators = []
    for a_cfg in anchor_cfg:
        
        anchor_generator = build_anchor_generator(a_cfg)
        anchor_generators.append(anchor_generator)

    similarity_calc = build_similarity_calculator(
        target_assigner_config.REGION_SIMILARITY_CALCULATOR)
    positive_fraction = target_assigner_config.SAMPLE_POSITIVE_FRACTION
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config.SAMPLE_SIZE)
    return target_assigner


def build_losses(loss_config):
    """Build losses based on the config.

    Builds classification, localization losses and optionally a hard example miner
    based on the config.

    Args:
    loss_config: A yaml.Loss object.

    Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.

    Raises:
    ValueError: If hard_example_miner is used with sigmoid_focal_loss.
    """
    
    classification_loss = _build_classification_loss(
        loss_config.classification_loss)
    localization_loss = _build_localization_loss(
        loss_config.localization_loss)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight
    hard_example_miner = None

    return (classification_loss, localization_loss,
            classification_weight,
            localization_weight, hard_example_miner)


def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

    Args:
    loss_config: A yaml.LocalizationLoss object.

    Returns:
    Loss based on the config.

    Raises:
    ValueError: On invalid loss_config.
    """

    if 'weighted_l2' in loss_config:
        config = loss_config.weighted_l2
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)

    if 'weighted_smooth_l1' in loss_config:
        config = loss_config.weighted_smooth_l1
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight)
    else:
        raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

    Args:
    loss_config: A yaml.ClassificationLoss object.

    Returns:
    Loss based on the config.

    Raises:
    ValueError: On invalid loss_config.
    """
    if  'weighted_sigmoid' in loss_config:
        return losses.WeightedSigmoidClassificationLoss()

    if 'weighted_sigmoid_focal' in loss_config:
        config = loss_config.weighted_sigmoid_focal
        # alpha = None
        # if config.HasField('alpha'):
        #   alpha = config.alpha
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(
                gamma=config.gamma,
                alpha=alpha)
    if 'weighted_softmax_focal' in loss_config :
        config = loss_config.weighted_softmax_focal
        # alpha = None
        # if config.HasField('alpha'):
        #   alpha = config.alpha
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SoftmaxFocalClassificationLoss(
            gamma=config.gamma,
            alpha=alpha)

    if 'weighted_softmax' in loss_config:
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(
            logit_scale=config.logit_scale)

    if 'bootstrapped_sigmoid' in loss_config:
        config = loss_config.bootstrapped_sigmoid
        return losses.BootstrappedSigmoidClassificationLoss(
            alpha=config.alpha,
            bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))
    else:
        raise ValueError('Empty loss config.')


def build_optimizer(optimizer_config, params, name=None):
    """Create optimizer based on config.
  Args:
    optimizer_config: A Optimizer proto message.
  Returns:
    An optimizer and a list of variables for summary.
  Raises:
    ValueError: when using an unsupported input data type.
  """

    if optimizer_config.name == 'rms_prop_optimizer':

        optimizer = paddle.optimizer.RMSProp(
            parameters = params,
            learning_rate=_get_base_lr_by_lr_scheduler(optimizer_config.learning_rate),
            rho=optimizer_config.decay,
            momentum=optimizer_config.momentum_optimizer_value,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay)

    if optimizer_config.name =='momentum_optimizer':

        optimizer = paddle.optimizer.SGD(
            parameters = params,
            learning_rate=_get_base_lr_by_lr_scheduler(optimizer_config.learning_rate),
            weight_decay=optimizer_config.weight_decay)

    if optimizer_config.name =='adam_optimizer':

        optimizer = paddle.optimizer.Adam(
            parameters = params,
            learning_rate=_get_base_lr_by_lr_scheduler(optimizer_config.learning_rate),
            weight_decay=optimizer_config.weight_decay)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_config.name)

    if optimizer_config.use_moving_average:
        raise ValueError('paddle don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_config.name
    else:
        optimizer.name = name
    return optimizer


def _get_base_lr_by_lr_scheduler(learning_rate_config):
    base_lr = None
    learning_rate_type = learning_rate_config.name 
    if learning_rate_type == 'constant_learning_rate':

        base_lr = learning_rate_config.learning_rate

    if learning_rate_type == 'exponential_decay_learning_rate':

        base_lr = learning_rate_config.initial_learning_rate

    if learning_rate_type == 'manual_step_learning_rate':

        base_lr = learning_rate_config.initial_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')

    if learning_rate_type == 'cosine_decay_learning_rate':

        base_lr = learning_rate_config.learning_rate_base
    if base_lr is None:
        raise ValueError(
            'Learning_rate %s not supported.' % learning_rate_type)

    return base_lr


def build_db_preprocess(cfg):
    prepors = []
    for k,v in cfg.items():
        if k == 'filter_by_min_num_points':
            prepor = DBFilterByMinNumPoint(v.min_num_point_pairs)
            prepors.append(prepor)
        elif k == 'filter_by_difficulty':
            prepor = DBFilterByDifficulty(v.removed_difficulties)
            prepors.append(prepor)
        else:
            raise ValueError("unknown database prep type")
    return prepors 


def build_dbsampler(db_sampler_cfg):
    cfg = db_sampler_cfg
    groups = cfg.sample_groups
    prepors = build_db_preprocess(cfg.database_prep_steps) # list 
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [g.name_to_max_num for g in groups]
    info_path = cfg.database_info_path 
    with open(info_path,'rb') as f:
        db_infos = pickle.load(f)
    if len(grot_range) == 0:
        grot_range =None 
    sampler = DataBaseSamplerV2(db_infos , groups, db_prepor,rate,grot_range)
    return sampler 


def build_dataset(input_reader_config,
                    model_config,
                    training,
                    voxel_generator,
                    target_assigner=None):
    """Builds a tensor dictionary based on the InputReader config.

    Returns:
        A tensor dict based on the input_reader_config.
    """
    generate_bev = model_config.POST_PROCESSING.use_bev
    without_reflectivity = model_config.WITHOUT_REFLECTIVITY
    num_point_features = model_config.NUM_POINT_FEATURES
    out_size_factor = model_config.BACKBONE.layer_strides[0] //model_config.BACKBONE.upsample_strides[0]
    cfg = input_reader_config
    db_sampler_cfg = input_reader_config.DATABASE_SAMPLER
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0:
        db_sampler = build_dbsampler(db_sampler_cfg)
    try:
        u_db_sampler_cfg = input_reader_config.UNLABELED_DATABASE_SAMPLER
        u_db_sampler = None 
        if len(u_db_sampler_cfg.sample_groups) > 0:
            u_db_sampler = build_dbsampler(u_db_sampler_cfg)
    except:
        u_db_sampler = None
    grid_size = voxel_generator.grid_size #[352,400]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    prep_func = partial(
        prep_pointcloud,
        root_path = cfg.KITTI_ROOT_PATH,
        class_names = cfg.CLASS_NAMES,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels = cfg.MAX_NUMBER_OF_VOXELS,
        remove_outside_points = False,
        create_targets = training,
        shuffle_points = cfg.SHUFFLE_POINTS,
        gt_rotation_noise = cfg.GROUNDTRUTH_ROTATION_UNIFORM_NOISE,
        gt_loc_noise_std = cfg.GROUNDTRUTH_LOCALIZATION_NOISE_STD,
        global_rotation_noise = cfg.GLOBAL_ROTATION_UNIFORM_NOISE,
        global_scaling_noise = cfg.GLOBAL_SCALING_UNIFORM_NOISE,
        global_loc_noise_std = (0.2, 0.2, 0.2),
        global_random_rot_range = cfg.GLOBAL_RANDOM_ROTATION_RANGE_PER_OBJECT,
        db_sampler = db_sampler,
        unlabeled_db_sampler = u_db_sampler,
        generate_bev = generate_bev,
        without_reflectivity=without_reflectivity,
        num_point_features=num_point_features,
        anchor_area_threshold=cfg.ANCHOR_AREA_THRESHOLD,
        gt_points_drop=cfg.GROUNDTRUTH_POINTS_DROP_PERCENTAGE,
        gt_drop_max_keep=cfg.GROUNDTRUTH_DROP_MAX_KEEP_POINTS,
        remove_points_after_sample=cfg.REMOVE_POINTS_AFTER_SAMPLE,
        remove_environment=cfg.REMOVE_ENVIRONMENT,
        use_group_id=False,
        out_size_factor=out_size_factor)
    dataset = KittiDataset(
        info_path = cfg.KITTI_INFO_PATH,
        root_path=cfg.KITTI_ROOT_PATH,
        num_point_features=num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func
    )
    return dataset 


def build_input_reader(input_reader_config,
                       model_config,
                       training,
                       voxel_generator,
                       target_assigner=None) -> DatasetWrapper:
    
    dataset = build_dataset(input_reader_config,
                            model_config,
                            training,
                            voxel_generator,
                            target_assigner)
    dataset = DatasetWrapper(dataset)
    return dataset 


def build_lr_schedules(optimizer_config, optimizer, last_step=-1):

    return _create_learning_rate_scheduler(optimizer_config.learning_rate, 
                                           optimizer, 
                                           last_step=last_step)


def _create_learning_rate_scheduler(learning_rate_config, optimizer, last_step=-1):
    """Create optimizer learning rate scheduler based on config.

    Args:
    learning_rate_config: A LearningRate proto message.

    Returns:
    A learning rate.

    Raises:
    ValueError: when using an unsupported input data type.
    """
    lr_scheduler = None
    learning_rate_type = learning_rate_config.name 
    if learning_rate_type == 'constant_learning_rate':  

        lr_scheduler = learning_schedules.Constant(
            optimizer, last_step=last_step)

    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config 
        lr_scheduler = learning_schedules.ExponentialDecay(
                optimizer, config.decay_steps, 
                config.decay_factor, config.staircase, last_step=last_step)

    if learning_rate_type == 'manual_step_learning_rate':
        config = learning_rate_config
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')
        learning_rate_step_boundaries = [x.step for x in config.schedule]
        learning_rate_sequence = [config.initial_learning_rate]
        learning_rate_sequence += [x.learning_rate for x in config.schedule]
        lr_scheduler = learning_schedules.ManualStepping(
            optimizer, learning_rate_step_boundaries, learning_rate_sequence, 
            last_step=last_step)

    if learning_rate_type == 'cosine_decay_learning_rate':
        config = learning_rate_config.cosine_decay_learning_rate
        lr_scheduler = learning_schedules.CosineDecayWithWarmup(
            optimizer, config.total_steps, 
            config.warmup_learning_rate, config.warmup_steps, 
            last_step=last_step)

    if lr_scheduler is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

    return lr_scheduler