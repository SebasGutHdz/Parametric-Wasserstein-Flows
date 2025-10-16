import jax
import jax.numpy as jnp
from flax import nnx
from typing import Dict, Any, Optional
import warnings
from  jaxtyping import PyTree,Array
from core.types import SampleArray,TimeArray,VelocityArray,TrajectoryArray
from architectures.node import NeuralODE
from architectures.architectures import MLP,ResNet
class ParametricModel(nnx.Module):
    """
    General class for parametric models in neural optimal transport.
    
    Supports pushforward maps T_θ: R^d -> R^d where ρ_t = (T_θ(t))#λ
    for transport between probability densities via learned maps.
    """

    def __init__(
        self,
        parametric_map: str = "node",
        architecture: list = [2, 1, 128], 
        activation_fn: str = "tanh",
        key: jax.random.PRNGKey = jax.random.key(0),
        ref_density: str = 'gaussian',
        scale_factor: float = 1e-3,
        **kwargs
    ):
        """
        Initialize parametric model for neural optimal transport.
        
        Args:
            parametric_map: Transport map type {'node', 'resnet', 'mlp'}
            architecture: [input_dim, num_layers, width_layers]
            activation_fn: Activation function for neural networks
            key: JAX random key for parameter initialization
            ref_density: Reference density {'gaussian', 'uniform'}
            scale_factor: Scale factor for initial parameters
            **kwargs: Additional arguments (see _validate_kwargs)
        """
        self.parametric_map = parametric_map
        self.problem_dimension = architecture[0]
        self.ref_density = ref_density
        self.scale_factor = scale_factor
        
        # Validate inputs
        self._validate_inputs(parametric_map, architecture, ref_density, kwargs)
        
        # Initialize model based on parametric map type
        self._initialize_model(architecture, activation_fn, key, kwargs)

    def _validate_inputs(self, parametric_map: str, architecture: list, 
                        ref_density: str, kwargs: Dict[str, Any]) -> None:
        """Validate input parameters and kwargs."""
        # Validate parametric map type
        valid_maps = {"node", "resnet", "mlp"}
        if parametric_map not in valid_maps:
            raise ValueError(f"parametric_map must be one of {valid_maps}, got {parametric_map}")
        
        # Validate architecture
        if len(architecture) != 3 or any(not isinstance(x, int) or x < 0 for x in architecture):
            raise ValueError("architecture must be [input_dim, num_layers, width_layers] with positive integers")
        
        # Validate reference density
        valid_densities = {"gaussian", "uniform"}
        if ref_density not in valid_densities:
            raise ValueError(f"ref_density must be one of {valid_densities}, got {ref_density}")
        
        # Validate NODE-specific kwargs
        if parametric_map == "node":
            self._validate_node_kwargs(kwargs)

    def _validate_node_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate NODE-specific keyword arguments."""
        # Time dependency
        if 'time_dependent' in kwargs:
            if not isinstance(kwargs['time_dependent'], bool):
                raise ValueError("time_dependent must be boolean")
        
        # Solver
        if 'solver' in kwargs:
            valid_solvers = {'euler', 'heun'}
            if kwargs['solver'] not in valid_solvers:
                raise ValueError(f"solver must be one of {valid_solvers}")
        
        # Time step
        if 'dt0' in kwargs:
            if not isinstance(kwargs['dt0'], (float, int)) or kwargs['dt0'] <= 0:
                raise ValueError("dt0 must be a positive number")
        
        # RHS model type
        if 'rhs_model' in kwargs:
            valid_rhs = {'mlp', 'resnet'}
            if kwargs['rhs_model'] not in valid_rhs:
                raise ValueError(f"rhs_model must be one of {valid_rhs}")

    def _initialize_model(self, architecture: list, activation_fn: str, 
                         key: jax.random.PRNGKey, kwargs: Dict[str, Any]) -> None:
        """Initialize the appropriate model based on parametric_map type."""
        rngs = nnx.Rngs(key)
        din, num_layers, width_layers = architecture
        dout = din  # Transport maps are typically R^d -> R^d
        
        if self.parametric_map == "node":
            self._initialize_node_model(din, dout, num_layers, width_layers, 
                                      activation_fn, rngs, kwargs)
        elif self.parametric_map == "resnet":
            self._initialize_resnet(din, dout, num_layers, width_layers, 
                                  activation_fn, rngs, kwargs)
        else:  
            self._initialize_MLP(din, dout, num_layers, width_layers, 
                                  activation_fn, rngs, kwargs)
        
        

    def _initialize_node_model(self, din: int, dout: int, num_layers: int,
                             width_layers: int, activation_fn: str,
                             rngs: nnx.Rngs, kwargs: Dict[str, Any]) -> None:
        """Initialize Neural ODE model with proper parameter handling."""
        # Set defaults with user feedback
        time_dependent = kwargs.get('time_dependent', True)
        if 'time_dependent' not in kwargs:
            warnings.warn("time_dependent not specified for NODE, defaulting to True")
        
        solver = kwargs.get('solver', 'euler')
        if 'solver' not in kwargs:
            warnings.warn("solver not specified for NODE, defaulting to 'euler'")
        
        dt0 = kwargs.get('dt0', 0.1)
        if 'dt0' not in kwargs:
            warnings.warn("dt0 not specified for NODE, defaulting to 0.1")
        
        rhs_model_type = kwargs.get('rhs_model', 'mlp')
        
        # Adjust input dimension for time dependency
        rhs_din = din + 1 if time_dependent else din
        
        # Create RHS model for the Neural ODE
        if rhs_model_type == "mlp":
            rhs_model = MLP(din=rhs_din, num_layers=num_layers,
                          width_layers=width_layers, dout=dout,
                          activation_fn=activation_fn, rngs=rngs)
        elif rhs_model_type == "resnet":
            rhs_model = ResNet(din=rhs_din, num_layers=num_layers,
                             width_layers=width_layers, dout=dout,
                             activation_fn=activation_fn, rngs=rngs)
        
        # Store the Neural ODE as the primary model
        self.model = NeuralODE(
            dynamics_model=rhs_model,
            solver=solver,
            time_dependent=time_dependent,
            dt0=dt0
        )
        graphdef, params = nnx.split(self.model)
        # params = jax.tree.map(lambda p: p * self.scale_factor, params)
        def init_layer(p):
            if len(p.shape) == 2: # weight matrix
                return self.scale_factor * p
            else: # bias vector        
                return self.scale_factor * p
        params = jax.tree.map(init_layer, params)
        self.model = nnx.merge(graphdef, params)
        self.dynamics = self.model.dynamics
        #Store for the NODE solver to see
        # self.time_dependent = time_dependent

    def _initialize_resnet(self, din: int, dout: int, num_layers: int,
                             width_layers: int, activation_fn: str,
                             rngs: nnx.Rngs, kwargs: Dict[str, Any]) -> None:
        """
        Initialize ResNet model. To guarantee that the initial map is close to identity,
        we initialize the last layer weights and biases to be very small.
        """
        self.model = ResNet(din=din, num_layers=num_layers,
                          width_layers=width_layers, dout=dout,
                          activation_fn=activation_fn, rngs=rngs)
        # Initialize parameters to be almost zero. 
        # This guarantees that the initial transport map is close to identity map    
        graphdef, params = nnx.split(self.model)
        params = jax.tree.map(lambda p: p * self.scale_factor, params)
        self.model = nnx.merge(graphdef, params)

    def _initialize_MLP(self, din: int, dout: int, num_layers: int,
                             width_layers: int, activation_fn: str,
                             rngs: nnx.Rngs, kwargs: Dict[str, Any]) -> None:
        """
        Initialize MLP model. To guarantee that the initial map is close to identity,
        We initialize the matrix of the layers to be the identity and the biases to be very small.
        """
        self.model = MLP(din=din, num_layers=num_layers,
                          width_layers=width_layers, dout=dout,
                          activation_fn=activation_fn, rngs=rngs)
        # Initialize matrix of each layer to be identity and biases to be almost zero. 
        # This guarantees that the initial transport map is close to identity map    
        graphdef, params = nnx.split(self.model)

        def init_layer(p):
            if len(p.shape) == 2: # weight matrix
                return jnp.eye(p.shape[0],p.shape[1]) + self.scale_factor * jax.random.normal(rngs.next(), p.shape)
            else: # bias vector
                return self.scale_factor * jax.random.normal(rngs.next(), p.shape)
        params = jax.tree.map(init_layer, params)
        self.model = nnx.merge(graphdef, params)

    def __call__(self, samples: jnp.ndarray, params: Optional[PyTree] = None,history: Optional[bool] = False ) -> jnp.ndarray:
        """
        Apply the parametric transport map T_θ(samples).
        
        Args:
            samples: Input samples from reference density, shape (batch_size, dim)
            params: Model parameters
            history: Whether to return the full trajectory (only for NODE models)
            
        Returns:
            Transported samples T_θ(samples), shape (batch_size, dim)
        """
        # Split and merge parameters for functional programming style
        if params is not None:
            graphdef, _ = nnx.split(self.model)
            model = nnx.merge(graphdef, params)
        else:
            model = self.model
        if history and self.is_node_model:
            return model(samples,history=history)
        if history and not self.is_node_model:
            warnings.warn("History flag is only applicable for NODE models. Ignoring.")
            return model(samples),None
        return model(samples)

    def sampler(self, key: jax.random.PRNGKey, num_samples: int) -> jnp.ndarray:
        """
        Sample from the reference density λ.
        
        Args:
            key: JAX random key
            num_samples: Number of samples to generate
            
        Returns:
            Samples from reference density, shape (num_samples, problem_dimension)
        """
        shape = (num_samples, self.problem_dimension)
        
        if self.ref_density == 'gaussian':
            return jax.random.normal(key, shape)
        elif self.ref_density == 'uniform':
            return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
        else:
            # This should never happen due to validation, but included for safety
            raise NotImplementedError(f'Reference density {self.ref_density} not implemented')

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model configuration."""
        info = {
            'parametric_map': self.parametric_map,
            'problem_dimension': self.problem_dimension,
            'ref_density': self.ref_density
        }
        
        if self.parametric_map == 'node' and hasattr(self.model, 'time_dependent'):
            info.update({
                'time_dependent': self.model.time_dependent,
                'solver': self.model.solver,
                'dt0': self.model.dt0
            })
        
        return info

    @property 
    def is_node_model(self) -> bool:
        """Check if this is a Neural ODE model."""
        return self.parametric_map == "node"

    def pushforward_density(self, samples: jnp.ndarray, params: Any) -> jnp.ndarray:
        """
        Compute pushforward T_θ#λ by applying the transport map.
        
        This is the main operation for neural optimal transport where we
        transform samples from reference density λ via the map T_θ.
        
        Args:
            samples: Samples from reference density λ
            params: Model parameters θ
            
        Returns:
            Transformed samples representing the target density
        """
        return self(samples, params)
    
    def log_likelihood(self, t: TimeArray,xt: TrajectoryArray,log_prob_init: Optional[Array] = None,method: str = 'exact',params: Optional[PyTree] = None,log_trajectory: bool = False) -> Array:
        """
        Compute log-likelihood of samples under the pushforward density T_θ#λ.
        
        Args:
            samples: Samples from target density
            params: Model parameters θ
            
        Returns:
            Log-likelihood values for each sample
        """
        if not self.is_node_model:
            raise NotImplementedError("Log-likelihood computation is only implemented for NODE models.")
        return self.model.log_likelihood(t,xt,log_prob_init=log_prob_init,method=method,params=params,log_trajectory=log_trajectory)
    def score_function(self,t: TimeArray,xt: TrajectoryArray,score_init: Optional[Array] = None,method: str = 'exact',params: Optional[PyTree] = None,score_trajectory: bool = False):
        """
        Compute score function of the pushforward density T_θ#λ.
        
        Args:
            samples: Samples from target density
            params: Model parameters θ
            
        Returns:
            Score function values for each sample
        """
        if not self.is_node_model:
            raise NotImplementedError("Score function computation is only implemented for NODE models.")
        return self.model.score_function(t,xt,score_init=score_init,method=method,params=params,score_trajectory=score_trajectory)
    def pull_back(self,x: SampleArray,params: Optional[PyTree] = None,history: bool = False) -> SampleArray:
        """
        Pull back samples x through the Neural ODE to obtain z
        
        Args:
            x: Target samples, shape (batch_size, feature_dim)
            params: Parameters for the dynamics model (if None, uses current model params)
            
        Returns:
            z: Reference samples, shape (batch_size, feature_dim)
        """
        if not self.is_node_model:
            raise NotImplementedError("Score function computation is only implemented for NODE models.")
        if params is None:
            _, params = nnx.split(self.dynamics)
        return self.model(x, t_span=(1.0, 0.0), params=params, history=history)