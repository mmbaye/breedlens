require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json({ limit: '200mb' }));  // lots d'images base64 -> payload large
app.use(express.static('.'));

// ─── Modèles Ollama disponibles ───────────────────────────────────────────────
const VISION_MODEL = 'qwen3-vl:8b';
const OLLAMA_URL   = 'http://127.0.0.1:11434';

// ─── Route analyse Ollama ─────────────────────────────────────────────────────
app.post('/analyze', async (req, res) => {
  const { imageBase64, imageType, prompt } = req.body;

  const system = "You are an expert in remote sensing, agronomy, urbanism and satellite image analysis. You analyze images from Google Earth Pro, drones or satellites. Your responses are precise, structured in markdown, and adapted to African or tropical contexts when relevant. Always respond in the same language as the user's question.";

  try {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:  VISION_MODEL,
        system: system,
        prompt: prompt,
        images: [imageBase64],
        stream: true
      })
    });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.response) {
            res.write(`data: ${JSON.stringify({ text: parsed.response })}\n\n`);
          }
          if (parsed.done) {
            res.write('data: [DONE]\n\n');
          }
        } catch (e) {}
      }
    }

    res.end();

  } catch (err) {
    res.status(500).json({ error: 'Ollama unavailable: ' + err.message });
  }
});

// ─── Route détection YOLO (image unique) ──────────────────────────────────────
app.post('/detect', async (req, res) => {
  try {
    const response = await fetch('http://127.0.0.1:5001/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: 'YOLO server unavailable: ' + err.message });
  }
});

// ─── Route détection YOLO par lot ─────────────────────────────────────────────
app.post('/detect-batch', async (req, res) => {
  try {
    const response = await fetch('http://127.0.0.1:5001/detect-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: 'YOLO server unavailable: ' + err.message });
  }
});

// ─── Route analyse sorgho (image annotée YOLO + Ollama) ──────────────────────
app.post('/analyze-sorghum', async (req, res) => {
  const {
    annotatedImage, count, grains_total, mass_g,
    conf_mean, slope_used, face_factor, plant_density,
    yield_t_ha, class_counts, lang
  } = req.body;

  const isFr = lang === 'fr';

  const prompt = isFr ? `
Tu analyses une image de panicule de sorgho avec détection automatique de grains par YOLOv11.

**Données de détection :**
- Grains détectés (face visible uniquement) : ${count}
- Facteur de correction face : ${face_factor} (valeur par défaut — pas encore calibré manuellement)
- Grains totaux estimés : ${grains_total} (= ${count} × ${face_factor})
- Masse estimée : ${mass_g} g (slope : ${slope_used} g/grain)
- Confiance moyenne : ${(conf_mean * 100).toFixed(1)}%
- Densité de plantation : ${plant_density.toLocaleString()} plants/ha
- Rendement estimé : **${yield_t_ha} t/ha**
- Classes détectées : ${JSON.stringify(class_counts)}

**Note importante :** Le modèle YOLOv11 ne détecte qu'une seule face de la panicule. Le facteur ${face_factor} est une hypothèse par défaut. Ces estimations devront être validées par comptage manuel.

**Analyse agronomique demandée :**
1. Évalue la densité et la distribution des grains sur la face visible
2. Commente la qualité apparente des grains (taille, uniformité, forme)
3. Identifie toute anomalie visible (zones vides, grains manquants, signes de maladie ou stress)
4. Donne une interprétation agronomique du rendement estimé (${yield_t_ha} t/ha)
5. Formule des recommandations pour améliorer la précision de l'estimation
` : `
You are analyzing a sorghum panicle image with automatic grain detection by YOLOv11.

**Detection data:**
- Grains detected (visible face only): ${count}
- Face correction factor: ${face_factor} (default value — not yet manually calibrated)
- Estimated total grains: ${grains_total} (= ${count} × ${face_factor})
- Estimated mass: ${mass_g} g (slope: ${slope_used} g/grain)
- Mean confidence: ${(conf_mean * 100).toFixed(1)}%
- Plant density: ${plant_density.toLocaleString()} plants/ha
- Estimated yield: **${yield_t_ha} t/ha**
- Detected classes: ${JSON.stringify(class_counts)}

**Important note:** The YOLOv11 model detects only one face of the panicle. The factor ${face_factor} is a default assumption. These estimates should be validated by manual counting.

**Agronomic analysis requested:**
1. Evaluate the density and distribution of grains on the visible face
2. Comment on apparent grain quality (size, uniformity, shape)
3. Identify any visible anomalies (empty zones, missing grains, disease or stress signs)
4. Provide an agronomic interpretation of the estimated yield (${yield_t_ha} t/ha)
5. Formulate recommendations to improve estimation accuracy
`;

  try {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:  VISION_MODEL,
        prompt: prompt,
        images: [annotatedImage],
        stream: true
      })
    });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.response) {
            res.write(`data: ${JSON.stringify({ text: parsed.response })}\n\n`);
          }
          if (parsed.done) {
            res.write('data: [DONE]\n\n');
          }
        } catch (e) {}
      }
    }

    res.end();

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Démarrage ────────────────────────────────────────────────────────────────
app.listen(3001, '0.0.0.0', () => {
  console.log(`✅ mobileBreederVision (Ollama — ${VISION_MODEL}) running on http://localhost:3001`);
});
