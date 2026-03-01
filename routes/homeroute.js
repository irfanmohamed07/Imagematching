import express from "express";
import { isAuthenticated } from "../middleware/auth.js";
import multer from "multer";
import { exec } from "child_process";
import path from "path";
import { readFileSync, existsSync, mkdirSync } from "fs";

const router = express.Router();

// Ensure uploads directory exists
const uploadsDir = "uploads";
if (!existsSync(uploadsDir)) {
  mkdirSync(uploadsDir, { recursive: true });
}

// Configure Multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

// File filter for images only
const fileFilter = (req, file, cb) => {
  const allowedTypes = /jpeg|jpg|png|webp|gif/;
  const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
  const mimetype = allowedTypes.test(file.mimetype);

  if (extname && mimetype) {
    return cb(null, true);
  } else {
    cb(new Error('Only image files are allowed'));
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }
});

// Store pending matches for async processing
const pendingMatches = new Map();

router.get("/", (req, res) => {
  res.render("home");
});

// POST route - Upload and show scanning animation
router.post("/upload", isAuthenticated, upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  const matchId = Date.now().toString();
  const uploadedImagePath = path.join("uploads", req.file.filename);

  // Store the match request
  pendingMatches.set(matchId, {
    status: "processing",
    imagePath: uploadedImagePath,
    filename: req.file.filename,
    result: null
  });

  // Start ML processing in background
  processMatch(matchId, uploadedImagePath, req.file.filename);

  // Immediately show scanning animation
  res.render("scanning", {
    title: "Scanning - Crescent College",
    matchId: matchId,
    uploadedImage: `/uploads/${req.file.filename}`
  });
});

// API endpoint to check match status
router.get("/api/match-status/:matchId", (req, res) => {
  const matchId = req.params.matchId;
  const match = pendingMatches.get(matchId);

  if (!match) {
    return res.json({ status: "not_found" });
  }

  res.json({
    status: match.status,
    ready: match.status === "complete" || match.status === "error"
  });
});

// Results page - called when processing is complete
router.get("/results/:matchId", (req, res) => {
  const matchId = req.params.matchId;
  const match = pendingMatches.get(matchId);

  if (!match) {
    return res.redirect("/");
  }

  // Clean up
  pendingMatches.delete(matchId);

  if (match.status === "error" || match.status === "processing" || !match.result) {
    return res.render("result", {
      title: "Error - Crescent College",
      matchResult: "Error",
      matchScore: "N/A",
      uploadedImage: `/uploads/${match.filename}`,
      matchedImage: null,
      personInfo: {
        name: "N/A",
        rrn: "N/A",
        department: "N/A",
        year: "N/A",
        section: "N/A"
      },
      error: match.error || (match.status === "processing" ? "Processing timed out or is taking too long. Please try again." : "Processing failed"),
      matches: []
    });
  }

  // Return the result
  res.render("result", match.result);
});

// Background processing function
function processMatch(matchId, uploadedImagePath, filename) {
  let pythonCmd = "python"; // Fallback to global python

  if (existsSync(path.join("venv", "Scripts", "python.exe"))) {
    pythonCmd = path.join("venv", "Scripts", "python"); // Windows venv
  } else if (existsSync(path.join("venv", "bin", "python"))) {
    pythonCmd = path.join("venv", "bin", "python"); // Mac/Linux venv
  }

  exec(`"${pythonCmd}" match.py "${uploadedImagePath}"`, {
    maxBuffer: 1024 * 1024 * 10
  }, (error, stdout, stderr) => {
    if (stderr) {
      console.log("Python ML Info:", stderr);
    }

    const match = pendingMatches.get(matchId);
    if (!match) return;

    if (error) {
      console.error(`exec error: ${error}`);
      match.status = "error";
      match.error = "Error running ML matching";
      return;
    }

    try {
      const result = JSON.parse(stdout);

      if (result.error) {
        match.status = "complete";
        match.result = {
          title: "No Match - Crescent College",
          matchResult: "No Match",
          matchScore: "N/A",
          uploadedImage: `/uploads/${filename}`,
          matchedImage: null,
          personInfo: {
            name: "N/A",
            rrn: "N/A",
            department: "N/A",
            year: "N/A",
            section: "N/A"
          },
          error: result.error,
          matches: [],
          imageQuality: result.quality_score || null
        };
        return;
      }

      const bestMatch = result.matches && result.matches.length > 0
        ? result.matches[0]
        : null;

      const matchedImage = bestMatch && bestMatch.filename
        ? `/images/${bestMatch.filename}`
        : null;

      let bestMatchMetadata = {};
      try {
        const imageData = JSON.parse(readFileSync("image_data.json", "utf-8"));
        bestMatchMetadata = bestMatch && bestMatch.filename
          ? imageData[bestMatch.filename] || {}
          : {};
      } catch (e) {
        console.error("Error reading image_data.json:", e);
      }

      const confidence = bestMatch ? bestMatch.confidence : 0;
      let matchQuality = "No Match";
      if (confidence >= 0.85) matchQuality = "Excellent Match";
      else if (confidence >= 0.75) matchQuality = "Strong Match";
      else if (confidence >= 0.65) matchQuality = "Good Match";
      else if (confidence >= 0.55) matchQuality = "Possible Match";

      match.status = "complete";
      match.result = {
        title: "Match Results - Crescent College",
        matchResult: bestMatch
          ? `${matchQuality} (${(confidence * 100).toFixed(1)}%)`
          : "No match found",
        matchScore: bestMatch
          ? `${(confidence * 100).toFixed(1)}%`
          : "N/A",
        uploadedImage: `/uploads/${filename}`,
        matchedImage: matchedImage,
        personInfo: {
          name: bestMatch?.name || bestMatchMetadata.Name || "N/A",
          rrn: bestMatch?.roll_no || bestMatchMetadata.RRN || "N/A",
          department: bestMatch?.department || bestMatchMetadata.Department || "N/A",
          year: bestMatch?.year || bestMatchMetadata.Year || "N/A",
          section: bestMatch?.section || bestMatchMetadata.Section || "N/A"
        },
        matches: result.matches || [],
        totalMatches: result.total_matches || 0,
        error: null,
        imageQuality: result.uploaded_image_quality || null,
        modelInfo: result.model_info || null
      };

    } catch (e) {
      console.error("Failed to parse JSON from Python script:", e);
      match.status = "error";
      match.error = "Failed to process image";
    }
  });
}

export default router;
