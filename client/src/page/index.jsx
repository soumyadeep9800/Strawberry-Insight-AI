import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";

import "../css/AgriVision.css";

// ============================================================
// FASTAPI GATEWAY
// ============================================================

const GATEWAY_URL = "http://127.0.0.1:8000";

// Delay between completed live inference requests.
const LIVE_INTERVAL_MS = 300;

export default function AgriVisionScanner() {
  // ==========================================================
  // GENERAL STATE
  // ==========================================================

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [model, setModel] = useState("yolov8s");
  const [confidence, setConfidence] = useState(0.25);

  const [prediction, setPrediction] = useState(null);
  const [advisories, setAdvisories] = useState(null);

  const [isPredicting, setIsPredicting] = useState(false);
  const [isFetchingAI, setIsFetchingAI] = useState(false);

  const [error, setError] = useState("");

  // ==========================================================
  // CAMERA STATE
  // ==========================================================

  const [cameraActive, setCameraActive] = useState(false);
  const [liveDetection, setLiveDetection] = useState(null);
  const [isLiveDetecting, setIsLiveDetecting] = useState(false);

  // This is inference throughput, not physical camera FPS.
  const [liveFps, setLiveFps] = useState(0);

  // ==========================================================
  // REFS
  // ==========================================================

  const fileInputRef = useRef(null);
  const treatmentRef = useRef(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);

  const liveTimerRef = useRef(null);
  const liveBusyRef = useRef(false);
  const liveRunningRef = useRef(false);

  const animationFrameRef = useRef(null);

  const liveDetectionRef = useRef(null);

  const modelRef = useRef(model);
  const confidenceRef = useRef(confidence);

  // ==========================================================
  // SYNCHRONIZE REFS
  // ==========================================================

  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  useEffect(() => {
    confidenceRef.current = confidence;
  }, [confidence]);

  useEffect(() => {
    liveDetectionRef.current = liveDetection;
  }, [liveDetection]);

  // ==========================================================
  // AUTO SCROLL TO AI ADVISORY
  // ==========================================================

  useEffect(() => {
    if (!advisories?.length || !treatmentRef.current) {
      return;
    }

    const frame = requestAnimationFrame(() => {
      treatmentRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });

    return () => cancelAnimationFrame(frame);
  }, [advisories]);

  // ==========================================================
  // PREVIEW URL CLEANUP
  // ==========================================================

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // ==========================================================
  // STOP CAMERA
  // ==========================================================

  const stopCamera = useCallback(() => {
    liveRunningRef.current = false;

    if (liveTimerRef.current) {
      clearTimeout(liveTimerRef.current);
      liveTimerRef.current = null;
    }

    liveBusyRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setIsLiveDetecting(false);
    setLiveFps(0);

    setLiveDetection(null);
    liveDetectionRef.current = null;
  }, []);

  // ==========================================================
  // FILE UPLOAD
  // ==========================================================

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];

    if (!selected) {
      return;
    }

    stopCamera();

    setFile(selected);

    setPreviewUrl((previousUrl) => {
      if (previousUrl) {
        URL.revokeObjectURL(previousUrl);
      }

      return URL.createObjectURL(selected);
    });

    setPrediction(null);
    setAdvisories(null);
    setError("");

    // Allows selecting the same file again.
    e.target.value = "";
  };

  // ==========================================================
  // NORMAL IMAGE ANALYSIS
  // ==========================================================

  const runAnalysisForFile = useCallback(async (selectedFile) => {
    if (!selectedFile) {
      return;
    }

    setIsPredicting(true);
    setError("");
    setPrediction(null);
    setAdvisories(null);

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);
      formData.append("model", modelRef.current);
      formData.append(
        "confidence",
        confidenceRef.current
      );

      const response = await fetch(
        `${GATEWAY_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let errData = {};

        try {
          errData = await response.json();
        } catch {
          // Ignore invalid JSON.
        }

        throw new Error(
          errData.detail || "Analysis failed."
        );
      }

      const data = await response.json();

      setPrediction(data.disease_detection);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the Gateway API."
      );
    } finally {
      setIsPredicting(false);
    }
  }, []);

  // ==========================================================
  // NORMAL ANALYSIS BUTTON
  // ==========================================================

  const runAnalysis = async () => {
    if (!file || isPredicting) {
      return;
    }

    await runAnalysisForFile(file);
  };

  // ==========================================================
  // AI ADVISORY / RAG
  // ==========================================================

  const getAdvisory = async (predictionOverride = null) => {
    const currentPrediction =
      predictionOverride || prediction;

    if (
      !currentPrediction ||
      !currentPrediction.detections
    ) {
      return;
    }

    if (currentPrediction.detections.length === 0) {
      return;
    }

    setIsFetchingAI(true);
    setError("");

    try {
      const payload = {
        detections: currentPrediction.detections.map(
          (d) => ({
            disease: d.disease,
            confidence: parseFloat(d.confidence),
          })
        ),
      };

      const response = await fetch(
        `${GATEWAY_URL}/advisory/multiple`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) {
        let errData = {};

        try {
          errData = await response.json();
        } catch {
          // Ignore.
        }

        throw new Error(
          errData.detail ||
            "Failed to fetch AI advisory."
        );
      }

      const data = await response.json();

      setAdvisories(data.advisories);
    } catch (err) {
      setError(
        err.message ||
          "Failed to communicate with Advisory service."
      );
    } finally {
      setIsFetchingAI(false);
    }
  };

  // ==========================================================
  // ANNOTATED IMAGE
  // ==========================================================

  const getAnnotatedImageSrc = (imgData) => {
    if (!imgData) {
      return "";
    }

    return imgData.startsWith("data:image")
      ? imgData
      : `data:image/jpeg;base64,${imgData}`;
  };

  // ==========================================================
  // START CAMERA
  //
  // IMPORTANT:
  // We DON'T access videoRef here.
  //
  // The video does not exist until cameraActive becomes true.
  // ==========================================================

  const startCamera = async () => {
    setError("");

    try {
      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {
        throw new Error(
          "Camera access is not supported by this browser."
        );
      }

      // Stop previous camera if any.
      stopCamera();

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "environment",
            },
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
          },
          audio: false,
        });

      // Store stream FIRST.
      streamRef.current = stream;

      setLiveDetection(null);
      liveDetectionRef.current = null;

      setLiveFps(0);

      // This causes React to render the <video>.
      setCameraActive(true);

      // DO NOT access videoRef here.
      //
      // The effect below will attach the stream after
      // React has rendered the video element.
    } catch (err) {
      setCameraActive(false);

      if (err.name === "NotAllowedError") {
        setError(
          "Camera permission was denied. Please allow camera access and try again."
        );
      } else if (err.name === "NotFoundError") {
        setError(
          "No camera was found on this device."
        );
      } else if (err.name === "NotReadableError") {
        setError(
          "The camera is already being used by another application."
        );
      } else if (err.name === "SecurityError") {
        setError(
          "Camera access requires a secure browser context such as HTTPS or localhost."
        );
      } else {
        setError(
          err.message ||
            "Unable to access the camera."
        );
      }
    }
  };

  // ==========================================================
  // ATTACH CAMERA STREAM AFTER VIDEO IS RENDERED
  // ==========================================================

  useEffect(() => {
    if (!cameraActive) {
      return;
    }

    const video = videoRef.current;
    const stream = streamRef.current;

    if (!video || !stream) {
      return;
    }

    let cancelled = false;

    const attachStream = async () => {
      try {
        video.srcObject = stream;

        await video.play();

        if (cancelled) {
          return;
        }
      } catch (err) {
        if (!cancelled) {
          console.error(
            "Video playback error:",
            err
          );

          setError(
            "Camera opened, but the video preview could not start."
          );
        }
      }
    };

    attachStream();

    return () => {
      cancelled = true;
    };
  }, [cameraActive]);

  // ==========================================================
  // CAMERA CLEANUP ON UNMOUNT
  // ==========================================================

  useEffect(() => {
    return () => {
      liveRunningRef.current = false;

      if (liveTimerRef.current) {
        clearTimeout(liveTimerRef.current);
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(
          animationFrameRef.current
        );
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());
      }

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  // ==========================================================
  // CAPTURE VIDEO FRAME
  // ==========================================================

  const captureVideoFrame = () => {
    const video = videoRef.current;

    if (
      !video ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      return null;
    }

    const canvas =
      document.createElement("canvas");

    const maxWidth = 960;

    const scale =
      video.videoWidth > maxWidth
        ? maxWidth / video.videoWidth
        : 1;

    canvas.width = Math.round(
      video.videoWidth * scale
    );

    canvas.height = Math.round(
      video.videoHeight * scale
    );

    const ctx = canvas.getContext("2d");

    if (!ctx) {
      return null;
    }

    ctx.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    return canvas;
  };

  // ==========================================================
  // ANALYZE LIVE FRAME
  // ==========================================================

  const analyzeLiveFrame = useCallback(async () => {
    if (!liveRunningRef.current) {
      return;
    }

    if (liveBusyRef.current) {
      return;
    }

    const canvas = captureVideoFrame();

    if (!canvas) {
      return;
    }

    liveBusyRef.current = true;
    setIsLiveDetecting(true);

    const startedAt = performance.now();

    try {
      const blob =
        await new Promise((resolve) => {
          canvas.toBlob(
            resolve,
            "image/jpeg",
            0.72
          );
        });

      if (!blob) {
        throw new Error(
          "Could not capture camera frame."
        );
      }

      const formData = new FormData();

      formData.append(
        "file",
        blob,
        "live-camera.jpg"
      );

      formData.append(
        "model",
        modelRef.current
      );

      formData.append(
        "confidence",
        confidenceRef.current
      );

      const response = await fetch(
        `${GATEWAY_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let errData = {};

        try {
          errData =
            await response.json();
        } catch {
          // Ignore.
        }

        throw new Error(
          errData.detail ||
            "Live detection failed."
        );
      }

      const data =
        await response.json();

      if (!liveRunningRef.current) {
        return;
      }

      const detection =
        data.disease_detection;

      liveDetectionRef.current =
        detection;

      setLiveDetection(detection);

      const elapsed =
        performance.now() -
        startedAt;

      const fps =
        1000 /
        Math.max(elapsed, 1);

      // This is inference throughput.
      setLiveFps(
        Number(
          Math.min(30, fps).toFixed(1)
        )
      );
    } catch (err) {
      // Do not kill the live camera because
      // one backend request failed.
      console.error(
        "Live detection error:",
        err
      );
    } finally {
      liveBusyRef.current = false;
      setIsLiveDetecting(false);
    }
  }, []);

  // ==========================================================
  // LIVE DETECTION LOOP
  // ==========================================================

  const startLiveDetectionLoop =
    useCallback(() => {
      if (!cameraActive) {
        return;
      }

      // Prevent duplicate loops.
      if (liveRunningRef.current) {
        return;
      }

      liveRunningRef.current = true;

      const loop = async () => {
        if (!liveRunningRef.current) {
          liveTimerRef.current = null;
          return;
        }

        await analyzeLiveFrame();

        if (!liveRunningRef.current) {
          liveTimerRef.current = null;
          return;
        }

        liveTimerRef.current =
          setTimeout(
            loop,
            LIVE_INTERVAL_MS
          );
      };

      loop();
    }, [
      cameraActive,
      analyzeLiveFrame,
    ]);

  // ==========================================================
  // START LIVE DETECTION WHEN CAMERA IS READY
  // ==========================================================

  useEffect(() => {
    if (!cameraActive) {
      return;
    }

    // Give the video element a little time to
    // receive the stream.
    const timer = setTimeout(() => {
      if (
        cameraActive &&
        streamRef.current
      ) {
        startLiveDetectionLoop();
      }
    }, 150);

    return () => {
      clearTimeout(timer);

      liveRunningRef.current = false;

      if (liveTimerRef.current) {
        clearTimeout(
          liveTimerRef.current
        );

        liveTimerRef.current = null;
      }
    };
  }, [
    cameraActive,
    startLiveDetectionLoop,
  ]);

  // ==========================================================
  // LIVE OVERLAY DRAWING
  // ==========================================================

  useEffect(() => {
    if (!cameraActive) {
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) {
      return;
    }

    const draw = () => {
      const ctx =
        canvas.getContext("2d");

      if (!ctx) {
        return;
      }

      const rect =
        video.getBoundingClientRect();

      const displayWidth =
        rect.width;

      const displayHeight =
        rect.height;

      if (
        displayWidth <= 0 ||
        displayHeight <= 0 ||
        !video.videoWidth ||
        !video.videoHeight
      ) {
        animationFrameRef.current =
          requestAnimationFrame(draw);

        return;
      }

      const dpr =
        window.devicePixelRatio || 1;

      canvas.width =
        Math.round(
          displayWidth * dpr
        );

      canvas.height =
        Math.round(
          displayHeight * dpr
        );

      canvas.style.width =
        `${displayWidth}px`;

      canvas.style.height =
        `${displayHeight}px`;

      ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
      );

      ctx.clearRect(
        0,
        0,
        displayWidth,
        displayHeight
      );

      const detections =
        liveDetectionRef.current
          ?.detections || [];

      // ------------------------------------------------------
      // object-fit: cover transformation
      // ------------------------------------------------------

      const videoWidth =
        video.videoWidth;

      const videoHeight =
        video.videoHeight;

      const scale =
        Math.max(
          displayWidth / videoWidth,
          displayHeight / videoHeight
        );

      const renderedWidth =
        videoWidth * scale;

      const renderedHeight =
        videoHeight * scale;

      const offsetX =
        (displayWidth -
          renderedWidth) /
        2;

      const offsetY =
        (displayHeight -
          renderedHeight) /
        2;

      detections.forEach((d) => {
        if (!d.bbox) {
          return;
        }

        let [
          x1,
          y1,
          x2,
          y2,
        ] = d.bbox;

        x1 =
          x1 * scale +
          offsetX;

        y1 =
          y1 * scale +
          offsetY;

        x2 =
          x2 * scale +
          offsetX;

        y2 =
          y2 * scale +
          offsetY;

        // Completely outside visible area.
        if (
          x2 < 0 ||
          y2 < 0 ||
          x1 > displayWidth ||
          y1 > displayHeight
        ) {
          return;
        }

        x1 = Math.max(
          0,
          Math.min(displayWidth, x1)
        );

        y1 = Math.max(
          0,
          Math.min(displayHeight, y1)
        );

        x2 = Math.max(
          0,
          Math.min(displayWidth, x2)
        );

        y2 = Math.max(
          0,
          Math.min(displayHeight, y2)
        );

        const width =
          x2 - x1;

        const height =
          y2 - y1;

        if (
          width <= 2 ||
          height <= 2
        ) {
          return;
        }

        // --------------------------------------------------
        // Bounding box
        // --------------------------------------------------

        ctx.strokeStyle =
          "#a8e063";

        ctx.lineWidth = 3;

        ctx.shadowColor =
          "rgba(168,224,99,0.55)";

        ctx.shadowBlur = 8;

        ctx.strokeRect(
          x1,
          y1,
          width,
          height
        );

        ctx.shadowBlur = 0;

        // --------------------------------------------------
        // Label
        // --------------------------------------------------

        const disease =
          d.disease || "Unknown";

        const score =
          Number(
            d.confidence || 0
          ) * 100;

        const label =
          `${disease} ${score.toFixed(1)}%`;

        ctx.font =
          "600 13px DM Sans, sans-serif";

        const textWidth =
          ctx.measureText(label).width;

        const labelWidth =
          textWidth + 18;

        const labelHeight = 28;

        const labelY =
          Math.max(
            0,
            y1 - labelHeight
          );

        ctx.fillStyle =
          "rgba(8, 12, 9, 0.94)";

        ctx.fillRect(
          x1,
          labelY,
          labelWidth,
          labelHeight
        );

        ctx.fillStyle =
          "#a8e063";

        ctx.fillText(
          label,
          x1 + 9,
          labelY + 19
        );

        // --------------------------------------------------
        // Detection point
        // --------------------------------------------------

        ctx.fillStyle =
          "#eef5e9";

        ctx.beginPath();

        ctx.arc(
          x1,
          y1,
          4,
          0,
          Math.PI * 2
        );

        ctx.fill();
      });

      animationFrameRef.current =
        requestAnimationFrame(draw);
    };

    animationFrameRef.current =
      requestAnimationFrame(draw);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current =
          null;
      }
    };
  }, [cameraActive]);

  // ==========================================================
  // CAPTURE & ANALYZE EXACT CAMERA FRAME
  // ==========================================================

  const captureAndAnalyze = async () => {
    if (isPredicting) {
      return;
    }

    const canvas =
      captureVideoFrame();

    if (!canvas) {
      setError(
        "Camera frame is not ready yet."
      );

      return;
    }

    try {
      const blob =
        await new Promise(
          (resolve) => {
            canvas.toBlob(
              resolve,
              "image/jpeg",
              0.9
            );
          }
        );

      if (!blob) {
        throw new Error(
          "Could not capture camera frame."
        );
      }

      const capturedFile =
        new File(
          [blob],
          `strawberry-camera-${Date.now()}.jpg`,
          {
            type: "image/jpeg",
          }
        );

      const imageUrl =
        URL.createObjectURL(
          capturedFile
        );

      setFile(capturedFile);

      setPreviewUrl(
        (previousUrl) => {
          if (previousUrl) {
            URL.revokeObjectURL(
              previousUrl
            );
          }

          return imageUrl;
        }
      );

      // IMPORTANT:
      // Don't use the last liveDetection.
      // Analyze the exact captured frame.
      setPrediction(null);
      setAdvisories(null);
      setError("");

      stopCamera();

      await runAnalysisForFile(
        capturedFile
      );
    } catch (err) {
      setError(
        err.message ||
          "Failed to capture and analyze camera frame."
      );
    }
  };

  // ==========================================================
  // RESET
  // ==========================================================

  const resetScanner = () => {
    stopCamera();

    setFile(null);

    setPreviewUrl(
      (previousUrl) => {
        if (previousUrl) {
          URL.revokeObjectURL(
            previousUrl
          );
        }

        return null;
      }
    );

    setPrediction(null);
    setAdvisories(null);
    setError("");
  };

  // ==========================================================
  // COUNTS
  // ==========================================================

  const detectionCount =
    prediction?.detections?.length || 0;

  const liveDetectionCount =
    liveDetection?.detections?.length || 0;

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="agri-app">

      {/* ====================================================
          NAVBAR
      ==================================================== */}

      <nav className="agri-navbar">

        <div className="agri-brand">

          <div className="agri-brand-icon">
            <span>✦</span>
          </div>

          <div>
            <div className="agri-brand-name">
              Agri-Tech Vision System
            </div>

            <div className="agri-brand-sub">
              Intelligent strawberry diagnostics
            </div>
          </div>

        </div>

        <div className="agri-nav-status">

          <span
            className={`agri-status-dot ${
              cameraActive
                ? "camera-online"
                : isPredicting ||
                  isFetchingAI
                ? "processing"
                : prediction
                ? "online"
                : ""
            }`}
          />

          {cameraActive
            ? "LIVE CAMERA"
            : isPredicting
            ? "ANALYZING"
            : isFetchingAI
            ? "AI CONSULTING"
            : prediction
            ? "ANALYSIS COMPLETE"
            : "SYSTEM READY"}

        </div>

      </nav>

      <main className="agri-main">

        {/* ==================================================
            HERO
        ================================================== */}

        <section className="agri-hero">

          <div className="agri-hero-copy">

            <div className="agri-eyebrow">
              <span />
              AI-POWERED CROP HEALTH
            </div>

            <h1>
              See what's happening
              <br />
              <span>inside your crop.</span>
            </h1>

            <p>
              Scan strawberry plants using a
              live camera or upload an image.
              Computer vision identifies disease
              regions, while RAG-powered AI
              generates context-aware advisory.
            </p>

          </div>

          <div className="agri-hero-stat">

            <div className="agri-stat-number">
              AI
            </div>

            <div>
              <strong>
                Vision Engine
              </strong>

              <span>
                YOLOv8s / RT-DETR
              </span>
            </div>

          </div>

        </section>

        {/* ==================================================
            ERROR
        ================================================== */}

        {error && (
          <div className="agri-error">

            <div className="agri-error-icon">
              !
            </div>

            <div>
              <strong>
                Something went wrong
              </strong>

              <p>
                {error}
              </p>
            </div>

          </div>
        )}

        {/* ==================================================
            WORKSPACE
        ================================================== */}

        <section className="agri-workspace">

          {/* =================================================
              SIDEBAR
          ================================================= */}

          <aside className="agri-sidebar">

            {/* MODEL */}

            <div className="agri-card">

              <div className="agri-card-heading">

                <div>
                  <span className="agri-section-number">
                    01
                  </span>

                  <h2>
                    Detection engine
                  </h2>
                </div>

              </div>

              <p className="agri-card-description">
                Select the computer vision model
                used for disease detection.
              </p>

              <div className="agri-models">

                <button
                  type="button"
                  className={
                    model === "yolov8s"
                      ? "agri-model active"
                      : "agri-model"
                  }
                  onClick={() =>
                    setModel("yolov8s")
                  }
                >

                  <div className="agri-model-top">

                    <span className="agri-model-icon">
                      Y
                    </span>

                    {model === "yolov8s" && (
                      <span className="agri-selected">
                        ✓
                      </span>
                    )}

                  </div>

                  <strong>
                    YOLOv8s
                  </strong>

                  <span>
                    Lightweight detector
                  </span>

                </button>

                <button
                  type="button"
                  className={
                    model === "rtdetr"
                      ? "agri-model active"
                      : "agri-model"
                  }
                  onClick={() =>
                    setModel("rtdetr")
                  }
                >

                  <div className="agri-model-top">

                    <span className="agri-model-icon">
                      R
                    </span>

                    {model === "rtdetr" && (
                      <span className="agri-selected">
                        ✓
                      </span>
                    )}

                  </div>

                  <strong>
                    RT-DETR
                  </strong>

                  <span>
                    Transformer detector
                  </span>

                </button>

              </div>

            </div>

            {/* CONFIDENCE */}

            <div className="agri-card">

              <div className="agri-card-heading">

                <div>
                  <span className="agri-section-number">
                    02
                  </span>

                  <h2>
                    Sensitivity
                  </h2>
                </div>

                <span className="agri-confidence-value">
                  {Math.round(
                    confidence * 100
                  )}
                  %
                </span>

              </div>

              <p className="agri-card-description">
                Adjust the minimum confidence
                required before reporting a
                detection.
              </p>

              <div className="agri-range-wrapper">

                <input
                  className="agri-range"
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={confidence}
                  onChange={(e) =>
                    setConfidence(
                      parseFloat(
                        e.target.value
                      )
                    )
                  }
                />

                <div className="agri-range-labels">
                  <span>
                    More detections
                  </span>

                  <span>
                    Higher certainty
                  </span>
                </div>

              </div>

            </div>

            {/* CAMERA */}

            <div className="agri-card agri-camera-control-card">

              <div className="agri-card-heading">

                <div>
                  <span className="agri-section-number">
                    03
                  </span>

                  <h2>
                    Live scanner
                  </h2>
                </div>

                {cameraActive && (
                  <span className="agri-camera-mini-status">
                    ● LIVE
                  </span>
                )}

              </div>

              <p className="agri-card-description">
                Open your camera and detect
                strawberry diseases directly
                on the live screen.
              </p>

              {!cameraActive ? (
                <button
                  type="button"
                  className="agri-camera-button"
                  onClick={startCamera}
                >
                  <span className="agri-camera-button-icon">
                    ◉
                  </span>

                  <span>
                    Open live camera
                  </span>

                  <span>
                    →
                  </span>
                </button>
              ) : (
                <button
                  type="button"
                  className="agri-camera-stop"
                  onClick={stopCamera}
                >
                  <span>
                    ■
                  </span>

                  Stop camera
                </button>
              )}

            </div>

            {/* GALLERY */}

            <div className="agri-card">

              <div className="agri-card-heading">

                <div>
                  <span className="agri-section-number">
                    04
                  </span>

                  <h2>
                    Crop sample
                  </h2>
                </div>

              </div>

              <div
                className={`agri-upload ${
                  file
                    ? "has-file"
                    : ""
                }`}
                onClick={() =>
                  fileInputRef.current?.click()
                }
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" ||
                    e.key === " "
                  ) {
                    e.preventDefault();

                    fileInputRef.current?.click();
                  }
                }}
              >

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={
                    handleFileChange
                  }
                  accept="image/*"
                  hidden
                />

                {file ? (
                  <>
                    <div className="agri-upload-preview">

                      <img
                        src={previewUrl}
                        alt="Selected crop"
                      />

                      <div className="agri-upload-check">
                        ✓
                      </div>

                    </div>

                    <div className="agri-upload-file">
                      {file.name}
                    </div>

                    <span>
                      Click to replace sample
                    </span>
                  </>
                ) : (
                  <>
                    <div className="agri-upload-icon">
                      ↑
                    </div>

                    <strong>
                      Upload crop image
                    </strong>

                    <span>
                      JPG, PNG or WEBP
                    </span>
                  </>
                )}

              </div>

              <button
                type="button"
                className="agri-analyze-button"
                onClick={runAnalysis}
                disabled={
                  !file ||
                  isPredicting
                }
              >

                <span>
                  {isPredicting
                    ? "Analyzing crop..."
                    : "Run AI diagnosis"}
                </span>

                {!isPredicting && (
                  <span className="agri-button-arrow">
                    →
                  </span>
                )}

              </button>

              {(file ||
                prediction) && (
                <button
                  type="button"
                  className="agri-reset-button"
                  onClick={resetScanner}
                >
                  Reset scanner
                </button>
              )}

            </div>

          </aside>

          {/* =================================================
              CONTENT
          ================================================= */}

          <div className="agri-content">

            {/* =================================================
                LIVE CAMERA
            ================================================= */}

            <div className="agri-card agri-live-camera-card">

              <div className="agri-content-header">

                <div>
                  <span className="agri-section-number">
                    05
                  </span>

                  <h2>
                    Live vision scanner
                  </h2>

                  <p>
                    Real-time disease localization
                    using your selected detection model.
                  </p>
                </div>

                {cameraActive && (
                  <div className="agri-live-camera-badge">
                    <span />
                    CAMERA ACTIVE
                  </div>
                )}

              </div>

              {!cameraActive ? (
                <div className="agri-camera-empty">

                  <div className="agri-camera-empty-icon">
                    ◉
                  </div>

                  <h3>
                    Camera scanner ready
                  </h3>

                  <p>
                    Open the live camera from
                    the control panel to see
                    disease areas directly
                    on your screen.
                  </p>

                  <button
                    type="button"
                    className="agri-camera-open-large"
                    onClick={startCamera}
                  >
                    Open camera
                    <span>
                      →
                    </span>
                  </button>

                </div>
              ) : (
                <>

                  <div className="agri-camera-stage">

                    <video
                      ref={videoRef}
                      className="agri-camera-video"
                      autoPlay
                      muted
                      playsInline
                    />

                    <canvas
                      ref={canvasRef}
                      className="agri-camera-overlay"
                    />

                    <div className="agri-camera-corner top-left" />
                    <div className="agri-camera-corner top-right" />
                    <div className="agri-camera-corner bottom-left" />
                    <div className="agri-camera-corner bottom-right" />

                    <div className="agri-camera-top-bar">

                      <div className="agri-camera-live-indicator">
                        <span />
                        LIVE DETECTION
                      </div>

                      <div className="agri-camera-model-indicator">
                        {model.toUpperCase()}
                      </div>

                    </div>

                    <div className="agri-camera-bottom-bar">

                      <div className="agri-camera-stats">

                        <span>
                          {liveDetectionCount}{" "}
                          {liveDetectionCount === 1
                            ? "disease"
                            : "diseases"}
                        </span>

                        <span>
                          {liveFps || "--"}{" "}
                          DETECTION FPS
                        </span>

                        <span>
                          {Math.round(
                            confidence * 100
                          )}
                          % THRESHOLD
                        </span>

                      </div>

                      {isLiveDetecting && (
                        <div className="agri-live-processing">
                          SCANNING
                        </div>
                      )}

                    </div>

                  </div>

                  <div className="agri-camera-actions">

                    <button
                      type="button"
                      className="agri-capture-button"
                      onClick={
                        captureAndAnalyze
                      }
                      disabled={
                        isPredicting
                      }
                    >

                      <span>
                        ◎
                      </span>

                      {isPredicting
                        ? "Analyzing..."
                        : "Capture & Analyze"}

                    </button>

                    <button
                      type="button"
                      className="agri-camera-stop-small"
                      onClick={
                        stopCamera
                      }
                    >
                      Stop camera
                    </button>

                  </div>

                </>
              )}

            </div>

            {/* =================================================
                IMAGE ANALYSIS
            ================================================= */}

            <div className="agri-card agri-image-card">

              <div className="agri-content-header">

                <div>
                  <span className="agri-section-number">
                    06
                  </span>

                  <h2>
                    Vision analysis
                  </h2>

                  <p>
                    Compare the original sample with
                    detected disease regions.
                  </p>
                </div>

                {prediction && (
                  <div className="agri-detection-badge">
                    <span />

                    {detectionCount}{" "}
                    {detectionCount === 1
                      ? "finding"
                      : "findings"}
                  </div>
                )}

              </div>

              {!previewUrl ? (
                <div className="agri-empty-view">

                  <div className="agri-empty-orbit">
                    <div>
                      ⌁
                    </div>
                  </div>

                  <h3>
                    Waiting for a crop sample
                  </h3>

                  <p>
                    Upload an image or capture
                    a frame from the live camera
                    to begin analysis.
                  </p>

                </div>
              ) : (
                <div className="agri-image-grid">

                  <div className="agri-image-container">

                    <div className="agri-image-label">

                      <span>
                        ORIGINAL
                      </span>

                      <small>
                        SOURCE IMAGE
                      </small>

                    </div>

                    <img
                      src={previewUrl}
                      alt="Original crop sample"
                    />

                  </div>

                  {prediction?.annotated_image ? (
                    <div className="agri-image-container detected">

                      <div className="agri-image-label">

                        <span>
                          AI DETECTION
                        </span>

                        <small>
                          {prediction.model}
                        </small>

                      </div>

                      <img
                        src={getAnnotatedImageSrc(
                          prediction.annotated_image
                        )}
                        alt="Annotated detection output"
                      />

                    </div>
                  ) : (
                    <div className="agri-image-placeholder">

                      {isPredicting ? (
                        <>
                          <div className="agri-scanning">
                            <span />
                            <span />
                            <span />
                          </div>

                          <strong>
                            Processing image
                          </strong>

                          <p>
                            Vision model is
                            scanning the sample...
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="agri-placeholder-icon">
                            ◌
                          </div>

                          <strong>
                            Detection output
                          </strong>

                          <p>
                            Run diagnosis to see
                            detected regions.
                          </p>
                        </>
                      )}

                    </div>
                  )}

                </div>
              )}

            </div>

            {/* =================================================
                TELEMETRY
            ================================================= */}

            {prediction && (
              <div className="agri-card">

                <div className="agri-content-header compact">

                  <div>
                    <span className="agri-section-number">
                      07
                    </span>

                    <h2>
                      Inference telemetry
                    </h2>
                  </div>

                  <span className="agri-live-label">
                    ● LIVE READING
                  </span>

                </div>

                <div className="agri-metrics">

                  <div className="agri-metric">

                    <span>
                      INFERENCE TIME
                    </span>

                    <strong>
                      {prediction.inference_time_ms != null
                        ? Number(
                            prediction.inference_time_ms
                          ).toFixed(2)
                        : "--"}

                      <small>
                        {" "}ms
                      </small>
                    </strong>

                    <div className="agri-metric-line" />

                  </div>

                  <div className="agri-metric">

                    <span>
                      ENERGY USED
                    </span>

                    <strong>
                      {prediction.energy_joules != null
                        ? Number(
                            prediction.energy_joules
                          ).toFixed(4)
                        : "--"}

                      <small>
                        {" "}J
                      </small>
                    </strong>

                    <div className="agri-metric-line" />

                  </div>

                  <div className="agri-metric">

                    <span>
                      CARBON FOOTPRINT
                    </span>

                    <strong>
                      {prediction.carbon_footprint_gco2e != null
                        ? Number(
                            prediction.carbon_footprint_gco2e
                          ).toFixed(6)
                        : "--"}

                      <small>
                        {" "}gCO₂e
                      </small>
                    </strong>

                    <div className="agri-metric-line" />

                  </div>

                </div>

              </div>
            )}

            {/* =================================================
                FINDINGS
            ================================================= */}

            {prediction && (
              <div className="agri-card">

                <div className="agri-content-header">

                  <div>
                    <span className="agri-section-number">
                      08
                    </span>

                    <h2>
                      Detection findings
                    </h2>

                    <p>
                      Disease signatures identified
                      by the selected vision model.
                    </p>
                  </div>

                  {!advisories &&
                    prediction.detections?.length > 0 && (
                      <button
                        type="button"
                        className="agri-ai-button"
                        onClick={() =>
                          getAdvisory()
                        }
                        disabled={
                          isFetchingAI
                        }
                      >

                        <span>
                          {isFetchingAI
                            ? "Preparing AI Advisory..."
                            : "Get AI Advisory"}
                        </span>

                        {!isFetchingAI && (
                          <span>
                            ✦
                          </span>
                        )}

                        {isFetchingAI && (
                          <span className="agri-ai-spinner" />
                        )}

                      </button>
                    )}

                </div>

                {prediction.detections?.length > 0 ? (
                  <div className="agri-findings">

                    {prediction.detections.map(
                      (d, idx) => (
                        <div
                          key={idx}
                          className="agri-finding"
                        >

                          <div className="agri-finding-index">
                            {String(
                              idx + 1
                            ).padStart(2, "0")}
                          </div>

                          <div className="agri-finding-main">

                            <div className="agri-finding-title">

                              <h3>
                                {d.disease}
                              </h3>

                              <span>
                                {(
                                  Number(
                                    d.confidence
                                  ) * 100
                                ).toFixed(1)}
                                %
                              </span>

                            </div>

                            <div className="agri-confidence-track">

                              <span
                                style={{
                                  width: `${Math.min(
                                    100,
                                    Math.max(
                                      0,
                                      Number(
                                        d.confidence
                                      ) * 100
                                    )
                                  )}%`,
                                }}
                              />

                            </div>

                            {d.bbox && (
                              <div className="agri-bbox">

                                <span>
                                  BOUNDING BOX
                                </span>

                                <code>
                                  [
                                  {d.bbox
                                    .map(
                                      (n) =>
                                        Number(
                                          n
                                        ).toFixed(1)
                                    )
                                    .join(", ")}
                                  ]
                                </code>

                              </div>
                            )}

                          </div>

                        </div>
                      )
                    )}

                  </div>
                ) : (
                  <div className="agri-no-disease">

                    <div className="agri-no-disease-icon">
                      ✓
                    </div>

                    <div>
                      <strong>
                        No disease signatures detected
                      </strong>

                      <p>
                        The selected model did not
                        find any detection above
                        the current confidence
                        threshold.
                      </p>
                    </div>

                  </div>
                )}

              </div>
            )}

            {/* =================================================
                AI TREATMENT
            ================================================= */}

            {advisories && (
              <div
                ref={treatmentRef}
                className="agri-card agri-treatment-card"
              >

                <div className="agri-treatment-heading">

                  <div>

                    <div className="agri-ai-title">
                      <span>
                        ✦
                      </span>

                      AI AGRONOMIST
                    </div>

                    <h2>
                      Treatment strategy
                    </h2>

                    <p>
                      Context-aware recommendations
                      generated from the detected
                      crop conditions.
                    </p>

                  </div>

                </div>

                <div className="agri-advisories">

                  {advisories.map(
                    (adv, idx) => (
                      <article
                        key={idx}
                        className="agri-advisory"
                      >

                        <div className="agri-advisory-top">

                          <div className="agri-advisory-number">
                            {String(
                              idx + 1
                            ).padStart(2, "0")}
                          </div>

                          <div>

                            <h3>
                              {adv.disease}
                            </h3>

                            {adv.confidence !== null &&
                              adv.confidence !==
                                undefined && (
                                <span>
                                  Detection confidence{" "}
                                  {(
                                    Number(
                                      adv.confidence
                                    ) * 100
                                  ).toFixed(2)}
                                  %
                                </span>
                              )}

                          </div>

                        </div>

                        {adv.advisory?.summary && (
                          <div className="agri-treatment-summary">

                            <span>
                              OVERVIEW
                            </span>

                            <p>
                              {
                                adv.advisory
                                  .summary
                              }
                            </p>

                          </div>
                        )}

                        <div className="agri-advisory-columns">

                          {adv.advisory?.symptoms?.length > 0 && (
                            <div className="agri-advisory-section">

                              <div className="agri-advisory-label">
                                <span>
                                  01
                                </span>

                                Symptoms
                              </div>

                              <ul>
                                {adv.advisory.symptoms.map(
                                  (
                                    symptom,
                                    i
                                  ) => (
                                    <li key={i}>
                                      {symptom}
                                    </li>
                                  )
                                )}
                              </ul>

                            </div>
                          )}

                          {adv.advisory?.prevention?.length > 0 && (
                            <div className="agri-advisory-section">

                              <div className="agri-advisory-label">
                                <span>
                                  02
                                </span>

                                Prevention
                              </div>

                              <ul>
                                {adv.advisory.prevention.map(
                                  (
                                    item,
                                    i
                                  ) => (
                                    <li key={i}>
                                      {item}
                                    </li>
                                  )
                                )}
                              </ul>

                            </div>
                          )}

                          {adv.advisory?.management?.length > 0 && (
                            <div className="agri-advisory-section">

                              <div className="agri-advisory-label">
                                <span>
                                  03
                                </span>

                                Management
                              </div>

                              <ul>
                                {adv.advisory.management.map(
                                  (
                                    item,
                                    i
                                  ) => (
                                    <li key={i}>
                                      {item}
                                    </li>
                                  )
                                )}
                              </ul>

                            </div>
                          )}

                          {adv.advisory?.hydroponic_considerations?.length > 0 && (
                            <div className="agri-advisory-section hydro">

                              <div className="agri-advisory-label">
                                <span>
                                  04
                                </span>

                                Hydroponic considerations
                              </div>

                              <ul>
                                {adv.advisory.hydroponic_considerations.map(
                                  (
                                    item,
                                    i
                                  ) => (
                                    <li key={i}>
                                      {item}
                                    </li>
                                  )
                                )}
                              </ul>

                            </div>
                          )}

                        </div>

                      </article>
                    )
                  )}

                </div>

              </div>
            )}

          </div>

        </section>

      </main>

      <footer className="agri-footer">

        <span>
          AGRIVISION / CROP INTELLIGENCE SYSTEM
        </span>

        <span>
          COMPUTER VISION · REAL-TIME DETECTION · RAG · AI ADVISORY
        </span>

      </footer>

    </div>
  );
}