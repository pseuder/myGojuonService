from hiraganajapanese import label

import coremltools as ct
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

scale = 1 / 255.0

# 加载 Keras 模型并确保在自定义作用域中注册 LeakyReLU 激活函数
with CustomObjectScope({"GlorotUniform": glorot_uniform(), "LeakyReLU": LeakyReLU()}):
    keras_model = load_model("./hiragana.h5")

# 指定具体的输入形状，假设批次大小为 1
input_shape = (1, 48, 48, 1)

# 设置输入类型
input_name = ct.ImageType(name="image", shape=input_shape, scale=scale)

# 使用 ClassifierConfig 设置分类标签
classifier_config = ct.ClassifierConfig(class_labels=label)

# 转换 Keras 模型到 Core ML 模型，指定源框架为 tensorflow
hiragana_model = ct.convert(
    keras_model,
    source="tensorflow",
    inputs=[input_name],
    classifier_config=classifier_config,
)

hiragana_model.author = "Aiyu Kamate"
hiragana_model.short_description = "Handwritten Hiragana Recognition"
hiragana_model.input_description["image"] = "Detects handwritten Hiragana"
hiragana_model.output_description["output"] = "Prediction of Hiragana"
hiragana_model.save("hiragana.mlmodel")
