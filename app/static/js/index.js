const fileInput = document.getElementById("fileInput");
const imagePreview = document.getElementById("imagePreview");

fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    
    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.width="600px";
            imagePreview.style.height="600px";
        };

        reader.readAsDataURL(file);
    } else {
        // Si el usuario cancela la selección, puedes mostrar un mensaje o restaurar la imagen predeterminada.
        imagePreview.src = "";
    }
});
