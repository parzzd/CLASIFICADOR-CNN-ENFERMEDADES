const imagePreview = document.getElementById("imagePreview");
const urlParams = new URLSearchParams(window.location.search);
const imageData = urlParams.get("image");

if (imageData) {
    imagePreview.src = decodeURIComponent(imageData);
} else {
    imagePreview.src = ""; // Si no se encuentra la imagen en la URL
}

