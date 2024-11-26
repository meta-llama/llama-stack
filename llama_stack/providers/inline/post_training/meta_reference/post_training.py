class MetaReferencePostTrainingImpl:
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config

    def supervised_fine_tune(
        self,
        job_uuid: str,
        model: str,
        dataset_id: str,
        validation_dataset_id: str,
        algorithm: FinetuningAlgorithm,
        algorithm_config: LoraFinetuningConfig,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = PostTrainingSFTRequest(
            job_uuid=job_uuid,
            model=model,
            dataset_id=dataset_id,
            validation_dataset_id=validation_dataset_id,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            logger_config=logger_config,
        )
        if request.algorithm == FinetuningAlgorithm.lora:
            recipe = LoraFinetuningRecipeSingleDevice(self.config, request)
            recipe.train()
        else:
            raise NotImplementedError()
