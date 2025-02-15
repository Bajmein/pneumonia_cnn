### **1. Configuración Inicial**
- [ ] Investigar tamaño óptimo para las imágenes
- [ ] Investigar tamaño óptimo para el batch
- [ ] Calcular media y desviación estándar directamente desde el dataset
- [ ] Revisar Hydra-Core
- [ ] Albumentations vs PyTorch Transforms
- [ ] Armar un método inteligente de rutas con pathlib

### **2. Diseño del Modelo**
- [ ] Utilizar torchsummary para obtener una descripción del modelo: https://pypi.org/project/torch-summary/
- [ ] Revisar PyTorch-Lightning
- [ ] Ver si me sirven convoluciones dilatadas
- [ ] De ser necesario, considerar dropout dinámico
- [ ] Investigar DropBlock
- [ ] Grad-CAM ++ y Score-CAM
- [ ] Visualizar características por cada capa

### **3. Entrenamiento**
- [ ] Utilizar Grad-CAM
- [ ] Utilizar Early Stopping (ver si se puede usar como decorador)
- [ ] Focal Loss vs función actual
- [ ] Revisar Optuna
- [ ] Meta-learning y contrastive learning
- [ ] Curvas de aprendizaje en gráfico, idealmente dinámicas
- [ ] Validación cruzada (10-fold cross-validation)
- [ ] Reducir Learning Rate on Plateau
- [ ] PyTorch Profiler
- [ ] Ver módulo FastProgress
- [ ] Tensorboard
- [ ] Wandb

### **4. Monitoreo y Métricas**
- [ ] Métricas como matriz de confusión y AUC-ROC
- [ ] Ver si AUC-ROC y AUC-ROC curva son lo mismo
- [ ] Decoradores para métricas y debug
- [ ] Ver qué onda SHAP
- [ ] Visualizar gradientes
- [ ] TorchMetrics, colormap
- [ ] Pytorch-Ignite

### **5. Optimización y Debugging**
- [ ] Ver qué onda el pruning
- [ ] Usar pdb para manejar tensores
- [ ] Manejar assert de tensores
- [ ] Visualizar tensores en Jupyter Notebook
- [ ] Generar métodos para guardar y cargar modelos
- [ ] Utilizar lru_cache donde se pueda
- [ ] Definir contextos con contextmanager donde se pueda
- [ ] Ver qué onda flake8

### **6. Generación de Datos**
- [ ] Considerar SMOTE para el desbalanceo
- [ ] Por qué utilizar SafeTensors

### **7. Validación y Evaluación Final**
- [ ] Comparar código actual con PyTorch-Lightning
- [ ] Comparar TorchMetrics y colormap
- [ ] Evaluar Pytorch-Ignite
- [ ] Explorar TorchSummaryX como complemento
- [ ] Realizar validación cruzada de 10 particiones

### **8. Visualización e Interpretabilidad**
- [ ] Implementar Grad-CAM y Grad-CAM++
- [ ] Visualizar mapas de características por capa
- [ ] Curvas de aprendizaje dinámicas
- [ ] Visualizar gradientes
- [ ] Utilizar Score-CAM para interpretabilidad

### **9. Gestión de Configuraciones y Rutas**
- [ ] Implementar rutas inteligentes con pathlib
- [ ] Revisar Hydra-Core para configuraciones avanzadas
- [ ] Utilizar dataclasses donde se pueda

### **10. Tareas Exploratorias**
- [ ] Investigar DropBlock
- [ ] Explorar meta-learning y contrastive learning
- [ ] Ver qué onda el pruning, aunque no parece necesario
- [ ] Revisar utilidad de SafeTensors
- [ ] Explorar SHAP para interpretabilidad avanzada
