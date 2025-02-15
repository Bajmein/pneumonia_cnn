from datetime import datetime


def create_model_filename(model, optimizer, criterion, scheduler, dropout, epochs, lr, layer_config):
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    criterion_name = criterion.__class__.__name__
    scheduler_name = scheduler.__class__.__name__ if scheduler is not None else "NoScheduler"

    # Si layer_config es una lista, la convertimos a cadena separada por guiones
    if isinstance(layer_config, list):
        layer_config_str = "-".join(map(str, layer_config))
        num_layers = len(layer_config)
    else:
        layer_config_str = layer_config
        num_layers = layer_config_str.count("-") + 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{model_name}_ep{epochs}_lr{lr}_{optimizer_name}_{criterion_name}_"
        f"{scheduler_name}_layers{num_layers}_{layer_config_str}_drop{dropout}_{timestamp}.pth"
    )
    return filename