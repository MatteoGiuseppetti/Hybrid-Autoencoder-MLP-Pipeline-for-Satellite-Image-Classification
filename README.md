# Hybrid Autoencoder–MLP Pipeline for Satellite Image Classification

Questo progetto implementa e valuta un Autoencoder Convoluzionale supervisionato per la classificazione di immagini satellitari. L'obiettivo principale è utilizzare l'encoder come estrattore di feature e addestrare un classificatore MLP sulle rappresentazioni latenti.

## Esplora il Progetto

Questo repository è organizzato in due sezioni per facilitare la consultazione: 

*  **[Leggi il Report Completo (No Lag)](Report/Hybrid_autoencoder–MLP_pipeline_for_satellite_image_classification.md)**
    * *Versione consigliata per la lettura:* include tutti i grafici, le analisi e i commenti in formato web veloce, senza dover scaricare nulla.

* **[Vai al Codice Sorgente (Jupyter)](Code/Hybrid_autoencoder–MLP_pipeline_for_satellite_image_classification.ipynb)**
    * *Versione per sviluppatori:* scarica o visualizza il file `.ipynb` originale per eseguire il codice.

## Obiettivi del Progetto
* Progettare un autoencoder convoluzionale per il dataset EuroSAT.
* Utilizzare lo spazio latente per addestrare un classificatore MLP.
* Valutare se lo spazio latente appreso fornisce feature discriminative per una classificazione accurata.

## Dataset
Il progetto utilizza il **dataset EuroSAT**, un benchmark per la classificazione land use/land cover basato su immagini satellitari Sentinel-2.
* **Contenuto:** Patch RGB di dimensioni $64\times64$ (GSD: 10 m/pixel).
* **Classi:** 10 classi (es. Foresta, Autostrada, Industriale, Colture annuali, ecc.).
* **Split:** 70% Training, 15% Validation, 15% Test.

## Architettura del Modello
Il modello è composto da due parti principali:
1.  **Encoder:** Una CNN a 4 blocchi che riduce la risoluzione spaziale e proietta l'input in uno spazio latente a **64 dimensioni**.
2.  **Classificatore MLP:** Una rete leggera collegata direttamente allo spazio latente (Input $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 10 classi).

L'architettura include anche un Decoder con convoluzioni trasposte per la ricostruzione dell'immagine, utilizzata durante il training supervisionato dell'Autoencoder.

## Risultati
Il classificatore MLP ha raggiunto un'accuratezza sul test set del **74.73%**.
* **Punti di forza:** Ottime performance su classi visivamente distinte come *Industrial*, *Residential* e *Highway*.
* **Criticità:** Confusione tra classi di vegetazione simili (es. *HerbaceousVegetation*, *Pasture*) e difficoltà specifica nella classe *Forest*, spesso confusa con *SeaLake*. 

## Autore
Matteo Giuseppetti - Università Cattolica del Sacro Cuore
