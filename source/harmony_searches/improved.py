from .original import OriginalHS


class ImprovedHS(OriginalHS):

    def __init__(self, par_min: float, par_max: float, **kwargs) -> None:
        super().__init__(par=0, **kwargs)
        self.par_min_ = par_min
        self.par_max_ = par_max
        del self.par_

    def _get_par(self, generation: int) -> float:
        return self.par_min_ + ((self.par_max_ - self.par_min_) * generation / self.n_generations_)
