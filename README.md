## Description

A multimodal approach was used, where an object detector identifies generalized classes (only two in total), followed by a classifier that compares similarities among cropped images.

After the classifier, precise coordinates of boxes are obtained on a 3x3 matrix. Lines are drawn from their centroids, and the point of intersection serves as a validator. The boxes are normalized to the first frame beforehand.

The first week was spent almost entirely on searching for an algorithmic approach. SIFT proved to be insufficiently robust for accomplishing the task with the required accuracy. However, it might be useful for cutting corners via crowdsourcing in the future.


## Описание

Использован мультимодельный подход, где детектор объектов выделяет обобщенные классы (всего два), а следующий за ним классификатор по кропам сверяет похожесть.

После классификатора получаем точные координаты боксов на 3х3 матрице, проводим линии из их центройдов, точка пересечения служит валидатором. Боксы предварительно нормализованы к первому фрейму.

Первая неделя была потрачена почти полностью на поиск алгоритмического подхода, SIFT показал себя недостаточно робастно для выполнения задания с необходимой точностью, однако им вполне можно срезать косты на краудсорсинг в будущем.


## Google Colab

https://colab.research.google.com/github/chiboreache/test_job_colab/blob/main/main.ipynb
