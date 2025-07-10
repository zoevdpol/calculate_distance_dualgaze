# Dual Gaze Distance Tool

Dit project is een uitbreiding op de open-source tool [gazeMapper](https://github.com/radicalbit/gazeMapper) en is ontwikkeld als onderdeel van een afstudeeronderzoek binnen het lectoraat *Betekenisvol Digitaal Innoveren*. De tool is gericht op het analyseren van gezamenlijke visuele aandacht in onderwijscontexten, bijvoorbeeld wanneer een docent en student samen naar een scherm kijken.

De applicatie maakt het mogelijk om:
- Gaze-data van **twee deelnemers** tegelijk te verwerken (dual gaze)
- Gaze-punten te **projecteren op een gezamenlijk schermvlak**
- De **afstand tussen de twee gaze-punten per frame** in millimeters te berekenen
- Een **video te genereren** waarin beide gaze-punten en hun onderlinge afstand worden weergegeven

Deze tool is uitsluitend getest met **Tobii Pro Glasses 3**, en is geoptimaliseerd voor gebruik in setups met ArUco-markers rondom een monitor.


## Installatie en gebruik

### 1. Download de applicatie
Ga naar de [Releases](https://github.com/zoevdpol/calculate_distance_dualgaze/releases) pagina van deze repository en download het nieuwste `.zip` bestand, bijvoorbeeld `dualgaze_app_windows.zip`.

### 2. Pak het bestand uit
Klik met de rechtermuisknop op het gedownloade `.zip` bestand en kies **Alles uitpakken**. Kies een locatie op je computer waar je de applicatie wilt gebruiken.

### 3. Start de applicatie
Open de uitgepakte map en dubbelklik op het bestand `start_dualgaze_app.bat`.  
De applicatie wordt automatisch gestart in een terminalvenster.

### 4. Problemen?
- Zorg dat je op een **Windows-systeem** werkt.
- Wordt het `.bat` bestand of Python geblokkeerd? Sta het bestand dan toe via je virusscanner of Windows Defender.
- Let op: het `.zip` bestand moet eerst **uitgepakt worden**. Het `.bat` bestand werkt niet als je de zip alleen opent zonder uit te pakken.

## Projectinformatie

Deze tool is ontwikkeld als afstudeerproject binnen het lectoraat Betekenisvol Digitaal Innoveren, gericht op het analyseren van gezamenlijke visuele aandacht met behulp van eye-tracking data.

## Stappenplan: Aanmaken en verwerken van een Dual Gaze-project in gazeMapper

Deze handleiding beschrijft hoe je stap voor stap een dual gaze-project opzet en verwerkt in gazeMapper, inclusief het importeren van data, uitvoeren van analyses en exporteren van resultaten.

### 1. Project aanmaken
- Open gazeMapper
- Selecteer in het startmenu de optie **“Make new dual gaze project”**
- Kies een lege map waarin het nieuwe project aangemaakt moet worden. De dual gaze-template wordt automatisch geladen.

### 2. Nieuwe sessie aanmaken
- Ga in het linker zijmenu naar het tabblad **“Session”**
- Klik op **“New session”**
- Geef de sessie een herkenbare naam en klik op **“Create session”**

### 3. Eye tracker recordings importeren
- Klik op **“Import eye tracker recordings”**
- Navigeer naar de mappen waarin de opnames van de Tobii Pro Glasses 3 staan
- Selecteer als type **“Tobii Pro Glasses 3”**, of een ander type indien van toepassing
- Sleep de gevonden opnames naar de juiste opname-namen in de sessie, bijvoorbeeld “lead” en “follow”
- De bijbehorende video’s worden automatisch geïmporteerd en gekoppeld aan de sessie

### 4. Automatische codering uitvoeren
- Klik met de rechtermuisknop op de sessie
- Selecteer **“Run auto codes”**

Deze pipeline voert automatisch de volgende stappen uit:
- Detectie van markers
- Automatische codering van synchronisatiepunten
- Automatische detectie van taaksegmenten

Let op:
- Wacht tot elk proces volledig is afgerond
- Je ziet in de terminal per stap een melding wanneer het proces voltooid is
- In de GUI verschijnen groene vinkjes of tellers (zoals 1/1, 2/2) zodra alles klaar is

Ga pas verder als alle stappen correct zijn uitgevoerd.

### 5. Handmatige episodecodering
- Klik met de rechtermuisknop op de sessie
- Kies **“Code episodes”**

Er opent nu een coderingsvenster waarin je validatie- en synchronisatie-episodes handmatig codeert.

#### Validatie-interval coderen:
Voer per opname de volgende stappen uit:
1. Zoek het begin van het validatie-interval in de video.
2. Pauzeer de video op het frame waarop de deelnemer begint te kijken naar het eerste fixatiepunt op de poster.
3. Druk op **`V`** om dit moment te markeren als begin van het interval.
4. Zoek het frame waarop de deelnemer het laatste fixatiepunt loslaat.
5. Druk opnieuw op **`V`** om het einde van het interval te markeren.
6. Herhaal dit voor extra validaties indien nodig.

Gebruik de GUI-functies zoals:
- `J` = frame vooruit
- `K` = frame terug
- `Spatiebalk` = pauze
- Tijdlijn om te zoeken

Druk op **Enter** om je codering op te slaan.

### 6. Eye tracker synchroniseren met scene camera (Sync ET to Cam)
- Klik met de rechtermuisknop op de sessie
- Kies **“Sync ET to Cam”**

Er opent een venster met twee tijdreeksen:
- Boven: gaze-data van de eye tracker
- Onder: camera-annotaties (bijv. markerherkenning)

#### Handmatige synchronisatie:
- Versleep de groene stip horizontaal om de tijdlijnen uit te lijnen
- De horizontale offset (in seconden) verschijnt rechtsonder
- Klik op **“Done”** als de synchronisatie klopt

### 7. Post-coding pipeline uitvoeren
- Klik opnieuw met de rechtermuisknop op de sessie
- Kies **“Run post coding”**

Deze stap voert automatisch de volgende processen uit:
- Synchronisatie met referentie
- Gaze mapping
- Validatie
- Afstandsberekening
- Genereren van outputvideo’s

### 8. Exporteren van de resultaten
- Klik met de rechtermuisknop op de sessie
- Kies **“Export trials”**
- De gaze-bestanden, validatieresultaten en gegenereerde video’s worden geëxporteerd naar de ingestelde map
