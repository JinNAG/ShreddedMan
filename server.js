/*
This is a java Script file that uses multer lib and express lib

-This file needs node.js to run JavaScript code outside of a web browser
-Express lib is a backend web application framework for node.js to help make web dev easier
-Multer lib is a node.js middleware for Express lib usedto handle multipart/form-data from an html <form> input

Most of the function is for type safety and file safety, the core of this file is to listen for any /upload tags sent from the web browser and then process the payload 

*/

const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { Readable } = require("node:stream");
const { pipeline } = require("node:stream/promises");

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

async function forwardBackend(endpoint, res, options = {}) {
    try {
        const response = await fetch(new URL(endpoint, BACKEND_URL), {
            ...options,
            signal: AbortSignal.timeout(60000)
        });
        res.status(response.status);
        res.set("Cache-Control", "no-store");
        for (const name of ["content-type", "content-disposition", "location", "retry-after"]) {
            const value = response.headers.get(name);
            if (value) res.set(name, value);
        }
        // Stream completed documents as PNG bytes, and status responses as JSON.
        if (response.body) {
            await pipeline(Readable.fromWeb(response.body), res);
        } else {
            res.end();
        }
    } catch (error) {
        console.error("Reconstruction service request failed:", error);
        if (res.headersSent) {
            res.destroy(error);
        } else {
            res.removeHeader("Content-Type");
            res.removeHeader("Content-Disposition");
            res.status(502).json({ detail: "The reconstruction service is unavailable. Please try again." });
        }
    }
}

// Make sure uploads directory exists
const uploadDirectory = path.join(__dirname, "uploads");

//Make sure backend process directory exists
const processDirectory = path.join(__dirname,"backend/img");

if (!fs.existsSync(uploadDirectory)) {
    fs.mkdirSync(uploadDirectory);
}

if(!fs.existsSync(processDirectory)){
    console.log("Backend folder for processing is missing");
    
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

// Only permit PNG and JPEG images
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 20 * 1024 * 1024
    },

    fileFilter: function (req, file, cb) {
        if (["image/png","image/jpeg"].includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error("Only PNG and JPEG images are allowed."));
        }
    }
});

// Serve index.html, CSS, images, etc.
app.use(express.static(path.join(__dirname, "public")));

// Handle your upload form
app.post("/upload", upload.array("ImageUpload", 20), async (req, res) => {
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ detail: "No images were uploaded." });
    }
    if (req.files.reduce((total, file) => total + file.size, 0) > 100 * 1024 * 1024) {
        return res.status(413).json({ detail: "The selected images exceed the 100 MiB submission limit." });
    }
    try {
        // All selected photos belong to one reconstruction submission.
        const formData = new FormData();
        for (const file of req.files) {
            const bytes = await fs.promises.readFile(file.path);
            formData.append("files", new Blob([bytes], { type: file.mimetype }), file.originalname);
        }
        formData.append("rotation", "auto");
        console.log("Uploaded:", req.files.map(file => file.filename));
        await forwardBackend("/api/submissions", res, { method: "POST", body: formData });
    } catch (error) {
        console.error("Could not read uploaded images:", error);
        res.status(500).json({ detail: "Could not read the uploaded images. Please try again." });
    }
});

app.get(["/api/submissions/:id", "/api/submissions/:id/document"], async (req, res) => {
    if (!/^[0-9a-f]{32}$/.test(req.params.id)) {
        return res.status(404).json({ detail: "Submission not found." });
    }
    const suffix = req.path.endsWith("/document") ? "/document" : "";
    await forwardBackend(`/api/submissions/${req.params.id}${suffix}`, res);
});

app.use("/upload", (error, req, res, next) => {
    const statusCode = error.code === "LIMIT_FILE_SIZE" ? 413 : 400;
    res.status(statusCode).json({ detail: error.message });
});

app.listen(PORT, () => {
console.log(`Server running on http://localhost:${PORT}`);
});
