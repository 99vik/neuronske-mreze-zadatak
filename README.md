# Programski zadatak iz kolegija "Neuronske mreže 1"

- U file-u replicirane_vjezbe.ipynb sam prošao kroz primjer manualnog forward i backward pass-a s prezentacija
- Zatim sam u zadatak.ipynb za iste težine i bias-e napravio petlju s forward i backward passom, optimizacijom i računanjem gubitka koji se u konačnici grafički prikazuje
- Na kraju sam na temelju prethodnog koda pretvorio jupyter notebook u python kod koji se može pokrenuti iz konzole, gdje se mogu odabrati hyper parametri i proizvoljno upisati ulazni i izlazni podaci u podaci.json file-u, težine i bias sakrivenog i izlaznog sloja imaju nasumične inicijalne vrijednosti

Za pokretanje main.py prvo je potrebno imati neke ulazne i izlazne podatke u podaci.json-u (po default-u su podaci za XOR problem), te se onda može pokrenuti kroz cli naredbom: "py main.py".
Moguće je direktno iz konzole unijeti odabrane hiperparametre, tako da se na prethodnu naredbu redoslijedom dodaju broj neurona u sakrivenom sloju,broj iteracija učenja, stopa/brzina učenja i momentum, npr. "py main.py 4 1000 0.1 0.5".
Ako se ne unesu dodatni argumenti uzimaju se default vrijednosti gdje je broj sakrivenih neurona 2, broj iteracija 500, brzina učenja 1, a momentum 0.9.
