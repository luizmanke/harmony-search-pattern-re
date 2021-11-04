import math
import numpy as np
from .original import OriginalHS


class ParameterAdaptiveHS(OriginalHS):

    def __init__(self, hmcr_min: float, hmcr_max: float, par_min: float, par_max: float, **kwargs) -> None:
        super().__init__(hmcr=0, par=0, **kwargs)
        self.hmcr_min_ = hmcr_min
        self.hmcr_max_ = hmcr_max
        self.par_min_ = par_min
        self.par_max_ = par_max
        del self.hmcr_
        del self.par_

    def _get_hmcr(self, generation: int) -> float:
        return self.hmcr_min_ + ((self.hmcr_max_ - self.hmcr_min_) * generation / self.n_generations_)

    def _get_par(self, generation: int) -> float:
        return self.par_max_ * math.exp(np.log(self.par_min_/self.par_max_) * generation / self.n_generations_)
