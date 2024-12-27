class Seed(object):
    """Class representing a single element of a corpus."""

    def __init__(self, root_seed, parent):
        """Inits the object.

        Args:
          cl: a transformation state to represent whether this seed has been
          coverage: a list to show the coverage
          root_seed: maintain the initial seed from which the current seed is sequentially mutated
          metadata: the prediction result
          ground_truth: the ground truth of the current seed

          l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
          between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1})  in Equation 2 of the paper
        Returns:
          Initialized object.
        """

        # self.metadata = metadata
        self.parent = parent
        self.root_seed = root_seed
        self.queue_time = None
        self.id = None
        # The initial probability to select the current seed.
        # self.probability = 0.8
        self.fuzzed_time = 0
