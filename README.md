### Problem
Na wejściu mamy daną sekwencję nukleotydów odpowiadającą genowi kodującemu pewne białko.
Na jej podstawie chcemy przewidzieć, jakie funkcje biologiczne będzie spełniało dane białko
### Model
Problem został ujęty jako predykcja tego, czy występują pewne krawędzie w grafie dwudzielnym (białka, funkcje)
Model na wyjściu zwraca embeddingi białek i funkcji, które następnie są przekazywane do klasyfikatora, który z kolei zwraca ‘prawdopodobieństwo’ tego, czy dane białko będzie spełniało daną funkcję
Zatem wejściem do modelu jest para (sekwencja nukleotydów, opis funkcji białka)

### Dane
Do przeprowadzenia treningu potrzebne są pliki:

    curl ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz --output data/goa_human.gaf.gz
    curl https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz --output data/uniprot_sprot.fasta.gz

pliki należy wypakować do folderu /data

### Trening
Aby rozpocząć jeden run treningowy, należy wywołać skrypt

    python3 train.py --config config/default.json

w pliku config/default.json znajdują się hiperparametry w formacie json

Aby rozpocząć gridsearch, należy uruchomić skrypt grid_search.py po zmianie przeszukiwanej przestrzeni parametrów w samym pliku