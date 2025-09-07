// Mostra anteprima appena l'utente sceglie un file
document.getElementById('imageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const preview = document.getElementById('originalImage');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
    }
});

document.addEventListener("DOMContentLoaded", () => {
    fetch("/legend")
        .then(res => res.json())
        .then(legend => {
            const container = document.getElementById("legendContainer");
            container.innerHTML = ""; // pulizia

            legend.forEach(item => {
                // wrapper della voce
                const entry = document.createElement("div");
                entry.classList.add("legend-item"); // usa il CSS

                // quadratino colorato
                const colorBox = document.createElement("div");
                colorBox.classList.add("legend-color");
                colorBox.style.backgroundColor = `rgb(${item.color[0]}, ${item.color[1]}, ${item.color[2]})`;

                // nome classe
                const label = document.createElement("span");
                label.textContent = item.name;

                // aggiungi al DOM
                entry.appendChild(colorBox);
                entry.appendChild(label);
                container.appendChild(entry);
            });
        })
        .catch(err => console.error("Errore nel caricamento della legenda:", err));
});


function handleFileUpload(event) {
    const file = event.target.files[0];
    const selectedModel = document.querySelector('input[name="model"]:checked').value;

    if (!file) return;

    console.log("Model chosen:", selectedModel);
    console.log("File uploaded:", file.name);

    // Mostra immagine originale subito
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('originalImage').src = e.target.result;
        document.getElementById('originalImage').style.display = 'block';
        document.getElementById('defaultOriginal').style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Mostra loader segmentata, nascondi immagini segmentate vecchie
    document.getElementById('loaderSegmented').style.display = 'flex';
    document.getElementById('segmentedImage').style.display = 'none';
    document.getElementById('defaultSegmented').style.display = 'none';

    // Invia immagine al backend
    const formData = new FormData();
    formData.append("model", selectedModel);
    formData.append('file', file);

    fetch('/segment', {
        method: 'POST',
        body: formData
    })
    .then(async res => {
        document.getElementById('loaderSegmented').style.display = 'none';
        if (!res.ok) {
            let err = await res.json();
            throw new Error(err.error || 'Server error');
        }
        return res.blob(); 
    })
    .then(blob => {
        const segmentedImage = document.getElementById('segmentedImage');
        const downloadLink = document.getElementById('downloadSegmented');

        const url = URL.createObjectURL(blob);
        segmentedImage.src = url;
        segmentedImage.style.display = 'block';

        downloadLink.href = url;
        downloadLink.style.display = 'inline-block';  // mostra il bottone download
    })
    .catch(err => {
        document.getElementById('loaderSegmented').style.display = 'none';
        alert('Errore: ' + err.message);
    });
}
