from typing import Optional, Sequence
import numpy as np
import scvi
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module import VAE

from src.modules.custom_module import My_VAE

class MyModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
            self,
            adata: AnnData,
            n_hidden: int,
            n_latent: int,
            **model_kwargs
    ):
        super().__init__(adata)

        self.module = VAE(
            n_input = self.summary_stats["n_vars"], # number of geners in input data
            n_hidden = n_hidden,
            n_batch = self.summary_stats["n_batch"], # number of batches, for example; differet experimaetnal setups, dates, labs, methods, ...
            n_latent = n_latent,
            **model_kwargs
        )

        self._model_summary_string = f"My VAE model with {n_latent} latent dimensions and {self.summary_stats.n_batch} batches(conditions/samples/whatever)"

        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        This method registers the fields of the AnnData object to be used in the model. It is called when the model is initialized with an AnnData object. It is also called when the model is loaded from a checkpoint, so it needs to be able to handle both cases.

        Parameters
        ----------
        adata
            The AnnData object to register.
        batch_key
            The key in `adata.obs` that corresponds to batch information. If `None`, no batch information will be registered.
        layer
            The key in `adata.layers` that corresponds to the input data. If `None`, `adata.X` will be used as input data.

        Returns
        -------
        Optional[AnnData]
            The AnnData object with registered fields. If `None`, the original AnnData object will be used.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]

        '''
            Ovu dummy fields su tu samo da zadovolje strukturu koju scvi-tools očekuje, ali obzirom da nemam JointObs 
            tj. matrice koje opisuju covariates to je jednsotavno prazno

            '''
    
        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,    
        )
        
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)



#--------------------------------------- Defining base SCVI model ----------------------------------------------------------------------------------------------

