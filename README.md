# Explicando Predicciones y Modelos de Machine Learning
# Taller impartido en el V Congreso de Usuarios de R en la Universidad de La Laguna (26/Nov/2026)
En los últimos años, se han desarrollado técnicas para la construcción de modelos predictivos y de machine learning muy complejas, en los cuales es muy difícil conocer qué variables afectan a las predicciones. Son llamadas “black boxes”, cajas negras. Por ejemplo: redes neuronales, ensemble methods (bagging, random forest, boosting), super vector machines, stacking methods, … Muchas de estas técnicas predicen muy bien, pero realmente se desconoce porqué. En este taller se recogen un conjunto de técnicas y herramientas para explorar, explicar y examinar estos modelos predictivos complejos. Son métodos agnósticos porque pueden ser aplicados a cualquier clase de modelo predictivo, ya que no es necesario conocer el modelo, ni cómo ha sido construido, lo único que interesa es saber cómo predice. Son técnicas que pretenden explorar las relaciones aprendidas por el modelo, y explicar los factores que influyen en las decisiones que toma el modelo. 

De forma general, se puede hablar de métodos de explicación local y global, según el objetivo. En el taller, se han elegido algunas de las técnicas más populares y que sean representativas de las diferentes aproximaciones que hay en este tipo de análisis, ya que es un campo de investigación muy activo actualmente. 

Los métodos de explicación locales, son los que explican la predicción hecha por el modelo para una única observación concreta. Se presentarán las siguientes técnicas: Break-down Plots, Shapley Values (SHAP), Local Interpretable Model Explanations (LIME) y los Ceteris-paribus Profiles. 

Los métodos de explicación globales, son los que explican las predicciones para un conjunto entero de datos, y en el taller se verán diferentes medidas para evaluar la Importancia de las Variables, los Partial dependence Profiles y los Local dependence and Accumulated-local Profiles.

Se usarán varios paquetes de R, entre ellos los más importantes que son el “iml” y el “DALEX”, y todas las técnicas se verán con un conjunto de datos, que nos ayude a entender las fortalezas y debilidades de cada aproximación.

