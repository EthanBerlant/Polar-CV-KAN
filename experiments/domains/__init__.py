"""
Domain modules for unified training script.

Each domain module exports:
- Trainer: BaseTrainer subclass with train_step/validate_step
- create_model(args): Model factory
- create_dataloaders(args): Data factory returning (train, val, test, info)
- DEFAULTS: Dict of domain-specific argument defaults
- add_args(parser): Function to add domain-specific arguments
"""

from .audio import (
    DEFAULTS as AUDIO_DEFAULTS,
)
from .audio import (
    AudioTrainer,
)
from .audio import (
    add_args as add_audio_args,
)
from .audio import (
    create_dataloaders as create_audio_dataloaders,
)
from .audio import (
    create_model as create_audio_model,
)
from .image import (
    DEFAULTS as IMAGE_DEFAULTS,
)
from .image import (
    ImageTrainer,
)
from .image import (
    add_args as add_image_args,
)
from .image import (
    create_dataloaders as create_image_dataloaders,
)
from .image import (
    create_model as create_image_model,
)
from .sst2 import (
    DEFAULTS as SST2_DEFAULTS,
)
from .sst2 import (
    SST2Trainer,
)
from .sst2 import (
    add_args as add_sst2_args,
)
from .sst2 import (
    create_dataloaders as create_sst2_dataloaders,
)
from .sst2 import (
    create_model as create_sst2_model,
)
from .synthetic import (
    DEFAULTS as SYNTHETIC_DEFAULTS,
)
from .synthetic import (
    SyntheticTrainer,
)
from .synthetic import (
    add_args as add_synthetic_args,
)
from .synthetic import (
    create_dataloaders as create_synthetic_dataloaders,
)
from .synthetic import (
    create_model as create_synthetic_model,
)
from .timeseries import (
    DEFAULTS as TIMESERIES_DEFAULTS,
)
from .timeseries import (
    TimeSeriesTrainer,
)
from .timeseries import (
    add_args as add_timeseries_args,
)
from .timeseries import (
    create_dataloaders as create_timeseries_dataloaders,
)
from .timeseries import (
    create_model as create_timeseries_model,
)

# Domain registry
DOMAINS = {
    "image": {
        "trainer": ImageTrainer,
        "create_model": create_image_model,
        "create_dataloaders": create_image_dataloaders,
        "defaults": IMAGE_DEFAULTS,
        "add_args": add_image_args,
    },
    "audio": {
        "trainer": AudioTrainer,
        "create_model": create_audio_model,
        "create_dataloaders": create_audio_dataloaders,
        "defaults": AUDIO_DEFAULTS,
        "add_args": add_audio_args,
    },
    "timeseries": {
        "trainer": TimeSeriesTrainer,
        "create_model": create_timeseries_model,
        "create_dataloaders": create_timeseries_dataloaders,
        "defaults": TIMESERIES_DEFAULTS,
        "add_args": add_timeseries_args,
    },
    "sst2": {
        "trainer": SST2Trainer,
        "create_model": create_sst2_model,
        "create_dataloaders": create_sst2_dataloaders,
        "defaults": SST2_DEFAULTS,
        "add_args": add_sst2_args,
    },
    "synthetic": {
        "trainer": SyntheticTrainer,
        "create_model": create_synthetic_model,
        "create_dataloaders": create_synthetic_dataloaders,
        "defaults": SYNTHETIC_DEFAULTS,
        "add_args": add_synthetic_args,
    },
}
