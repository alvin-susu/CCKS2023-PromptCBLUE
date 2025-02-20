from unittest.mock import Mock, MagicMock


def create_mock_trainer():
    # 创建虚拟 trainer
    mock_trainer = Mock()

    # 模拟 evaluate() 返回的 metrics
    mock_metrics = {"eval_loss": 0.5, "eval_accuracy": 0.85}
    mock_trainer.evaluate.return_value = mock_metrics

    # 模拟 predict() 返回的结果
    mock_predict_results = MagicMock()
    mock_predict_results.metrics = {"predict_loss": 0.4, "predict_accuracy": 0.9}
    mock_predict_results.predictions = [[1, 2, 3], [4, 5, 6]]  # 虚拟预测 token IDs
    mock_predict_results.label_ids = [[1, 2, 3], [4, 5, 6]]  # 虚拟标签 token IDs
    mock_trainer.predict.return_value = mock_predict_results

    # 模拟其他必要方法
    mock_trainer.log_metrics = Mock()
    mock_trainer.save_metrics = Mock()
    mock_trainer.is_world_process_zero = Mock(return_value=True)  # 确保主进程执行

    return mock_trainer