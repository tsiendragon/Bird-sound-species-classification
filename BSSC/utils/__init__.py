from typing import Callable

import timm

register_model: Callable[[Callable], Callable] = timm.models.registry.register_model
