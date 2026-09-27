const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;

// Make sure uploads directory exists
const uploadDirectory = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadDirectory)) {
    fs.mkdirSync(uploadDirectory);
}

// Configure how uploaded files are stored
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDirectory);
    },

    filename: function (req, file, cb) {
        const uniqueName =
        Date.now() + "-" + Math.round(Math.random() * 1E9) + ".png";
        cb(null, uniqueName);
    }
});

// Only permit PNG images
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 5 * 1024 * 1024
    },

    fileFilter: function (req, file, cb) {
        if (file.mimetype === "image/png") {
            cb(null, true);
        } else {
            cb(new Error("Only PNG images are allowed."));
        }
    }
});

// Serve index.html, CSS, images, etc.
app.use(express.static(path.join(__dirname, "public")));

// Handle your upload form
app.post("/upload", upload.single("ImageUpload"), (req, res) => {
    if (!req.file) {
        return res.status(400).send("No image was uploaded.");
    }

    console.log("Uploaded:", req.file.filename);
    res.status(200).send("Upload successful");
});

app.listen(PORT, () => {
console.log(`Server running on http://localhost:${PORT}`);
});