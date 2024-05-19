[![en](https://img.shields.io/badge/language-EN-blue.svg)](https://github.com/pzemla/Road-sign-images-generation-using-GAN/blob/main/README.md)

# Generacja obrazów znaków drogowych wykorzystując sieć GAN

**Zależności**

Python 3.9.13

matplotlib 3.8.3

notebook 7.1.2

numpy 1.24.1

pandas  2.2.1

scikit-learn 1.4.1.post1 Python 3.9.13

torch 2.2.1+cu118

**Jak uruchomić**
1. Pobierz dataset z https://www.kaggle.com/datasets/flo2607/traffic-signs-classification
2. Umieść folder myData w tym samym folderze co GAN.ipynb
3. Uruchom GAN.ipynb w Jupyter Notebook

## Przegląd
Celem tego projektu jest zbudowanie sieci neuronowej ‘generatywnych przeciwników’ (GAN -Generative Adversarial Network) do generowania obrazów znaków drogowych. GAN jest zaimplementowany przy użyciu Pythona z biblioteką Pytorch. Model jest szkolony na oznaczonym zbiorze danych zawierającym obrazy 42 różnych znaków drogowych o wymiarach 32x32x3 (zbiór danych zawiera 42 foldery ze zdjęciami znaków z różnych perspektyw). Niska rozdzielczość obrazów jest spowodowana ograniczonym czasem i mocą obliczeniową możliwą do przeznaczenia na trenowanie sieci. Projekt ten służy jako ćwiczenie edukacyjne i praktyczne zastosowanie technik głębokiego uczenia się do zadań generowania obrazów.

Poniżej są zdjęcia przykładowych obrazów:

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/792960b2-49cf-486b-a8ab-ed02187661fd)
![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/ac42f213-ca5c-47e7-8a3f-9a4e955e6ab2)

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/66345175-ee70-45e0-8584-4718746b4412)
![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/4881b98d-4607-4530-bee8-caff6c8547a5)

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/85d1c0f7-620a-45ce-9476-387fd9b9e14f)
![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/cf30cbb8-fc62-483b-8dc2-36df41ef23d8)

# Struktura sieci GAN

Sieć GAN składa się z dwóch ‘pomniejszych’ sieci – generatora i dyskryminatora. Zadaniem generatora w tym wypadku jest generacja realistycznych obrazów znaków drogowych, a dyskryminatora rozpoznanie, czy podawane na jego wejście obrazy są prawdziwe (pochodzą z datasetu) czy wygenerowane przez generator. Podczas treningu generator uczy się generować bardziej realistyczne obrazy, a dyskryminator lepiej je rozpoznawać. Wejściem generatora jest szum losowy, a dyskryminatora obraz (z generatora lub datasetu). Wyjściem generatora jest wygenerowany obraz, a dyskryminatora ocena czy obraz jest wygenerowany (0) czy prawdziwy (1).

**Generator:**

|Warstwa|Opis|Input|Output|
| ------------- | ------------- | ------------- | ------------- |
|View|Konwersja inputu z 3-wymiarowego na 1-wymiarowy|32x1x1|32|
|Linear|Warstwa liniowa|32|512|
|View|Konwersja inputu z 1-wymiarowego na 3-wymiarowy|512|512x1x1|
|Transposed convolution|Stride=4, kernel=1|512x1x1|256x4x4|
|Batch Normalization|Normalizacja outputu z transponowanej warstwy konwolucyjnej|256x4x4|256x4x4|
|ReLU|Funkcja aktywacji|256x4x4|256x4x4|
|Transposed convolution|Stride=4, kernel=2, padding=1|256x4x4|128x8x8|
|Batch Normalization|Normalizacja outputu z transponowanej warstwy konwolucyjnej|128x8x8|128x8x8|
|ReLU|Funkcja aktywacji|128x8x8|128x8x8|
|Transposed convolution|Stride=4, kernel=2, padding=1|128x8x8|64x16x16|
|Batch Normalization|Normalizacja outputu z transponowanej warstwy konwolucyjnej|64x16x16|64x16x16|
|ReLU|Funkcja aktywacji|64x16x16|64x16x16|
|Transposed convolution|Stride=4, kernel=2, padding=1|64x16x16|3x32x32|
|Tanh|Funkcja aktywacji|3x32x32|3x32x32|

**Dyskryminator**

|Warstwa|Opis|Input|Output|
| ------------- | ------------- | ------------- | ------------- |
|Convolutional|Stride=4, kernel=2, padding=1|3x32x32|64x16x16|
|Leaky ReLU|Funkcja aktywacji|64x16x16|64x16x16|
|Convolutional|Stride=4, kernel=2, padding=1|64x16x16|128x8x8|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|128x8x8|128x8x8|
|Leaky ReLU|Funkcja aktywacji|128x8x8|128x8x8|
|Convolutional|Stride=4, kernel=2, padding=1|128x8x8|256x4x4|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|256x4x4|256x4x4|
|Leaky ReLU|Funkcja aktywacji|256x4x4|256x4x4|
|Convolutional|Stride=4, kernel=1|256x4x4|1x1x1|
|Sigmoid|Funkcja aktywacji|1x1x1|1x1x1|

# Optymalizator i funkcja straty

Optymalizator – Adam (learning rate=0.001)
funkcja straty – Binary cross-entropy loss

Optymalizator Adam został wybrany ponieważ dynamicznie dostosowuje learning rate do każdego parametru podczas treningu, przez co nie trzeba dostosowywać malenia współczynnika uczenia (learning rate decay). Spośród innych optymalizatorów testowanych (Adagrad i RMSprop) zapewniał on najlepsze wyniki w datasecie testowym. Generator i dyskryminator mogą mieć różne optymalizatory, ale w tym wypadku w obu wykorzystywany jest Adam.

Funkcja straty binarnej entropii krzyżowej (binary cross-entropy - BCE) została wybrana ponieważ jej wynik może być interpretowany jako prawdopodobieństwo przynależności do jednej z dwóch klas, przez co jest ona często wykorzystywana w klasyfikacji prawda/fałsz. Generator i dyskryminator mogą mieć dwie różne funkcje strat, jednak tutaj w obu wypadkach wykorzystywane jest BCE, ponieważ w dyskryminatorze można je interpretować jako prawdopodobieństwo poprawnego rozpoznania obrazu, a w generatorze jako prawdopodobieństwo oszukania dyskryminatora przez wygenerowany obraz.

# Rezultaty

Przykładowe wygenerowane obrazy po określonej ilości epok treningu.

100 epoka

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/7dfe3797-8e39-4fc3-8ef7-2967a8f50ff7)

180 epoka

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/68631bcf-9b97-41d7-8837-2abfd123588d)

260 epoka

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/35fd3e61-8581-4151-aa1b-849d01e838ff)

300 epoka

![image](https://github.com/pzemla/Road-sign-images-generation-using-GAN/assets/135070990/5ccfc09d-23a9-41be-80c4-a33dea4f2585)

Wygenerowane obrazy znaków drogowych po wytrenowaniu generatora przypominają kształtem i kolorystyką prawdziwe obrazy znaków drogowych. W większości obrazów wnętrze znaku (gdzie powinien znajdować się obrazek lub tekst) jest rozmazane. Może to być spowodowane niską jakością obrazów w datasecie, zbyt duża liczba różniących się wnętrz znaków (co można rozwiązać poprzez zastosowanie sieci conditional GAN), wielkością sieci lub nieoptymalnymi ustawieniami parametrów. Część obrazów jest niemal niewidoczna, jest prawdopodobnie nie jest spowodowane przy treningu sieci neuronowej, tylko przez dataset który zawiera zdjęcia znaków robione w nocy. Tło jest rozmazane, ponieważ obrazy znaków mają wiele różnych rodzajów tła (białe tło, czarne/nocne tło, normalne tło, rozmazane tło), przez co dyskryminator się na nim nie skupia, więc generator nie uczy się tworzyć realistycznego tła.

## Licencja

Ten projekt jest dostępny na licencji MIT - zobacz plik LICENSE dla szczegółów.
