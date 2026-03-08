# WMA_LAB1

LAB1 – detekcja i śledzenie czerwonego obiektu w materiale wideo.

## Uruchomienie

```bash
pip install -r requirements.txt
```

Uruchom program, podając ścieżkę do pliku wideo:

```bash
python lab1_object_detection.py --video F1.MOV
```

Obserwacje do zadania:
- program wczytuje wideo z parametru `--video`, 
- segmentuje kolor czerwony w przestrzeni HSV (H zostało zmienione po probkowaniu na 5 przez szum na brązie),
- oczyszcza maskę operacjami morfologicznymi,
- wyświetla dwa okna: oryginalne wideo oraz maskę,
- oznacza wykryty obiekt okręgiem i wyświetla odchylenie w pikselach (prawa/lewa strona) w postaci paska oraz tekstu.
