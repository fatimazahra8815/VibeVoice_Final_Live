// Initialize copy buttons when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get all DOM elements
    const textInput = document.getElementById('textInput');
    const voiceSelect = document.getElementById('voiceSelect');
    const stepsRange = document.getElementById('stepsRange');
    const cfgRange = document.getElementById('cfgRange');
    const stepsVal = document.getElementById('stepsVal');
    const cfgVal = document.getElementById('cfgVal');
    const generateBtn = document.getElementById('generateBtn');
    const stopBtn = document.getElementById('stopBtn');
    const charCount = document.getElementById('charCount');
    const visualizer = document.getElementById('visualizer');
    const latencyVal = document.getElementById('latencyVal');
    const genTimeVal = document.getElementById('genTimeVal');
    const etaVal = document.getElementById('etaVal');
    const progressFill = document.getElementById('progressFill');
    const downloadBtn = document.getElementById('downloadBtn');
    const audioPlayer = document.getElementById('audioPlayer');
    const outputCard = document.getElementById('outputCard');

    let audioContext = null;
    let analyser = null;
    let ws = null;
    let isGenerating = false;
    let recordedChunks = [];
    let nextStartTime = 0;
    let animationId = null;
    let sourceFromPlayer = null;

    // Tab Switching Logic
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));

                tab.classList.add('active');
                const targetId = tab.dataset.tab + 'Tab';
                const target = document.getElementById(targetId);
                if (target) {
                    target.classList.add('active');
                    if (tab.dataset.tab === 'cloning') {
                        loadClonedVoices();
                    }
                }
            });
        });
    }

    // UI Updates
    if (textInput && charCount) {
        textInput.addEventListener('input', () => {
            charCount.textContent = `${textInput.value.length} characters`;
        });
    }
    if (stepsRange && stepsVal) {
        stepsRange.addEventListener('input', () => {
            stepsVal.textContent = stepsRange.value;
        });
    }
    if (cfgRange && cfgVal) {
        cfgRange.addEventListener('input', () => {
            cfgVal.textContent = cfgRange.value;
        });
    }

    // Load Config
    async function loadConfig() {
        try {
            const res = await fetch('/config');
            const data = await res.json();

            if (voiceSelect && data.voices) {
                const currentVoice = voiceSelect.value;
                voiceSelect.innerHTML = data.voices.map(v =>
                    `<option value="${v}" ${v === (currentVoice || data.default_voice) ? 'selected' : ''}>${v}</option>`
                ).join('');
            }
        } catch (e) {
            console.error('Failed to load config', e);
        }
    }
    loadConfig();

    // Generic Visualizer Logic
    function createVisualizer(canvasId, analyserNode) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !analyserNode) return null;

        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        let animationId = null;

        const resize = () => {
            if (canvas && canvas.parentElement) {
                const rect = canvas.parentElement.getBoundingClientRect();
                canvas.width = rect.width * window.devicePixelRatio;
                canvas.height = rect.height * window.devicePixelRatio;
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            }
        };

        resize();
        window.addEventListener('resize', resize);

        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animationId = requestAnimationFrame(draw);
            analyserNode.getByteFrequencyData(dataArray);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;

            ctx.clearRect(0, 0, w, h);

            const barWidth = (w / bufferLength) * 2;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * h;

                if (barHeight > 0) {
                    const gradient = ctx.createLinearGradient(0, h, 0, 0);
                    gradient.addColorStop(0, '#3b82f6');
                    gradient.addColorStop(1, '#60a5fa');

                    ctx.fillStyle = gradient;
                    ctx.fillRect(x, h - barHeight, barWidth - 1, barHeight);
                }
                x += barWidth;
            }
        }

        // Return control object
        return {
            start: () => { if (!animationId) draw(); },
            stop: () => { if (animationId) cancelAnimationFrame(animationId); animationId = null; ctx.clearRect(0, 0, canvas.width, canvas.height); }
        };
    }

    // Global references for visualizers
    let mainVisualizer = null;
    let podcastVisualizer = null;
    let tts15bVisualizer = null;
    let tts15bAudioContext = null;
    let tts15bAnalyser = null;

    // Audio Handling
    function initAudio() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;
            analyser.smoothingTimeConstant = 0.5;
            analyser.connect(audioContext.destination);

            mainVisualizer = createVisualizer('visualizer', analyser);

            // Link the audio player element to our visualizer context
            if (audioPlayer && !sourceFromPlayer) {
                try {
                    sourceFromPlayer = audioContext.createMediaElementSource(audioPlayer);
                    sourceFromPlayer.connect(analyser);
                } catch (e) {
                    console.warn('Could not create media element source:', e);
                }
            }
        }
        if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume();
        }
    }

    // Connect player events to visualizer
    if (audioPlayer) {
        audioPlayer.addEventListener('play', () => {
            initAudio();
            if (mainVisualizer) mainVisualizer.start();
        });
        audioPlayer.addEventListener('pause', () => {
            // Optional: Stop visualizer on pause to save resources, or keep it if we want to visualize silence
        });
    }

    async function playChunk(pcmData) {
        if (!audioContext) return;

        recordedChunks.push(new Int16Array(pcmData));

        const pcm16 = new Int16Array(pcmData);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768.0;
        }

        const buffer = audioContext.createBuffer(1, float32.length, 24000);
        buffer.getChannelData(0).set(float32);

        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(analyser);

        const startTime = Math.max(nextStartTime, audioContext.currentTime);
        source.start(startTime);
        nextStartTime = startTime + buffer.duration;

        if (mainVisualizer) mainVisualizer.start();
    }

    if (generateBtn) {
        generateBtn.addEventListener('click', () => {
            if (isGenerating) return;
            if (!textInput) {
                console.error('Text input element not found');
                return;
            }

            initAudio();
            isGenerating = true;
            recordedChunks = [];
            generateBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            if (downloadBtn) downloadBtn.disabled = true;
            if (outputCard) outputCard.style.opacity = "0.5";
            if (audioPlayer) audioPlayer.src = "";

            if (mainVisualizer) mainVisualizer.start();

            generateBtn.innerHTML = '<span class="btn-icon animate-spin">⏳</span> Synthesis...';
            if (latencyVal) latencyVal.textContent = 'Connecting...';
            if (audioContext) nextStartTime = audioContext.currentTime + 0.1;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.binaryType = 'arraybuffer';

            const startTime = Date.now();
            let firstChunkTime = null;
            let chunksReceived = 0;

            const estimatedTotalChunks = Math.max(10, textInput.value.length * 0.6);

            ws.onopen = () => {
                ws.send(JSON.stringify({
                    text: textInput ? textInput.value : "",
                    voice: voiceSelect ? voiceSelect.value : "",
                    cfg: cfgRange ? parseFloat(cfgRange.value) : 1.5,
                    steps: stepsRange ? parseInt(stepsRange.value) : 5
                }));
                if (latencyVal) latencyVal.textContent = 'Inferencing...';
                if (genTimeVal) genTimeVal.textContent = '0.0s';
                if (etaVal) etaVal.textContent = 'Estimating...';
            };

            const timerInterval = setInterval(() => {
                if (!isGenerating) {
                    clearInterval(timerInterval);
                    return;
                }
                if (genTimeVal) genTimeVal.textContent = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
            }, 100);

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'done') {
                        clearInterval(timerInterval);
                        if (etaVal) etaVal.textContent = 'Finished';
                        if (progressFill) progressFill.style.width = '100%';
                        finish(true);
                    } else if (msg.type === 'error') {
                        clearInterval(timerInterval);
                        finish(false);
                    }
                } else {
                    chunksReceived++;
                    if (!firstChunkTime) {
                        firstChunkTime = Date.now();
                        if (latencyVal) latencyVal.textContent = `${firstChunkTime - startTime}ms`;
                    }

                    const progress = Math.min(98, (chunksReceived / estimatedTotalChunks) * 100);
                    if (progressFill) progressFill.style.width = `${progress}%`;

                    if (chunksReceived > 3) {
                        const elapsedSinceFirst = (Date.now() - firstChunkTime) / 1000;
                        const speed = chunksReceived / elapsedSinceFirst;
                        const remainingChunks = Math.max(0, estimatedTotalChunks - chunksReceived);
                        let remainingTime = remainingChunks / speed;

                        if (etaVal) {
                            if (remainingTime < 0.2) {
                                etaVal.textContent = 'Finishing...';
                            } else {
                                etaVal.textContent = `${remainingTime.toFixed(1)}s`;
                            }
                        }
                    }

                    playChunk(event.data);
                }
            };

            ws.onclose = () => { if (isGenerating) finish(false); };
            ws.onerror = () => { if (isGenerating) finish(false); };
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (ws) ws.close();
            finish(false);
        });
    }

    function createWavBlob() {
        if (recordedChunks.length === 0) return null;

        const totalSamples = recordedChunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const dataLength = totalSamples * 2;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, 24000, true);
        view.setUint32(28, 24000 * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (const chunk of recordedChunks) {
            for (let i = 0; i < chunk.length; i++) {
                view.setInt16(offset, chunk[i], true);
                offset += 2;
            }
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const blob = createWavBlob();
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `vibevoice_${Date.now()}.wav`;
            link.click();
            URL.revokeObjectURL(url);
        });
    }

    function finish(wasSuccessful = false) {
        isGenerating = false;
        if (generateBtn) generateBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;

        if (wasSuccessful) {
            if (outputCard) outputCard.style.opacity = "1";
            if (downloadBtn) downloadBtn.disabled = false;
            const blob = createWavBlob();
            if (blob && audioPlayer) {
                audioPlayer.src = URL.createObjectURL(blob);
            }
        } else {
            if (downloadBtn) downloadBtn.disabled = true;
            if (outputCard) outputCard.style.opacity = "0.5";
        }

        if (generateBtn) generateBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Synthesis';
        if (ws) {
            ws.onclose = null;
            ws.onerror = null;
            ws.close();
        }
        if (etaVal && etaVal.textContent !== 'Finished') {
            etaVal.textContent = '--';
        }
    }

    /* -----------------------------------------------------------
       PODCAST LOGIC
       ----------------------------------------------------------- */
    const podcastScript = document.getElementById('podcastScript');
    const numSpeakersRange = document.getElementById('numSpeakersRange');
    const numSpeakersVal = document.getElementById('numSpeakersVal');
    const speakerSelectors = document.getElementById('speakerSelectors');
    const podcastCfgRange = document.getElementById('podcastCfgRange');
    const podcastCfgVal = document.getElementById('podcastCfgVal');
    const disableCloning = document.getElementById('disableCloning');
    const podcastGenerateBtn = document.getElementById('podcastGenerateBtn');
    const podcastStopBtn = document.getElementById('podcastStopBtn');
    const podcastLog = document.getElementById('podcastLog');
    const podcastOutputCard = document.getElementById('podcastOutputCard');
    const podcastDownloadBtn = document.getElementById('podcastDownloadBtn');
    const podcastPlayer = document.getElementById('podcastPlayer');

    // Voice cloning elements
    const dropZone = document.getElementById('dropZone');
    const audioUpload = document.getElementById('audioUpload');

    let podcastWs = null;
    let isPodcastGenerating = false;
    let podcastChunks = [];
    let podcastVoices = [];
    let podcastAudioContext = null;
    let podcastNextStartTime = 0;

    // Updates
    if (numSpeakersRange && numSpeakersVal) {
        numSpeakersRange.addEventListener('change', () => {
            numSpeakersVal.textContent = numSpeakersRange.value;
            updateSpeakerSelectors(parseInt(numSpeakersRange.value));
        });
    }

    if (podcastCfgRange && podcastCfgVal) {
        podcastCfgRange.addEventListener('input', () => {
            podcastCfgVal.textContent = podcastCfgRange.value;
        });
    }

    // Helper to build dropdowns
    function updateSpeakerSelectors(count) {
        if (!speakerSelectors) return;
        speakerSelectors.innerHTML = '';
        for (let i = 1; i <= count; i++) {
            const wrapper = document.createElement('div');
            wrapper.className = 'field';
            wrapper.style.marginBottom = '15px';

            const label = document.createElement('label');
            label.textContent = `Speaker ${i}`;

            const select = document.createElement('select');
            select.id = `speaker${i}`;
            select.className = 'speaker-select';

            if (podcastVoices.length > 0) {
                podcastVoices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v;
                    opt.textContent = v;
                    if (podcastVoices.indexOf(v) === (i - 1) % podcastVoices.length) {
                        opt.selected = true;
                    }
                    select.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.textContent = "Loading...";
                select.appendChild(opt);
            }

            wrapper.appendChild(label);
            wrapper.appendChild(select);
            speakerSelectors.appendChild(wrapper);
        }
    }

    async function loadPodcastConfig() {
        try {
            const res = await fetch('/podcast/config');
            const data = await res.json();
            if (data.voices) {
                podcastVoices = data.voices;
                if (numSpeakersRange) updateSpeakerSelectors(parseInt(numSpeakersRange.value));
            }
        } catch (e) {
            console.error("Failed to load podcast config", e);
        }
    }

    const randomScriptBtn = document.getElementById('randomScriptBtn');
    if (randomScriptBtn) {
        randomScriptBtn.addEventListener('click', loadRandomScript);
    }

    async function loadRandomScript() {
        try {
            const res = await fetch('/podcast/examples');
            const examples = await res.json();
            if (examples.length > 0) {
                const randomIdx = Math.floor(Math.random() * examples.length);
                const exRes = await fetch(`/podcast/example/${examples[randomIdx].index}`);
                const exData = await exRes.json();

                if (podcastScript) {
                    podcastScript.value = exData.script;
                }
                if (numSpeakersRange && numSpeakersVal && exData.speakers) {
                    numSpeakersRange.value = exData.speakers;
                    numSpeakersVal.textContent = exData.speakers;
                    updateSpeakerSelectors(exData.speakers);
                }
            } else {
                if (podcastScript) podcastScript.value = "No examples available on server. Please manually enter a script.";
            }
        } catch (e) {
            console.error("Failed to load random script", e);
            if (podcastScript) podcastScript.value = "Error loading example.";
        }
    }

    if (numSpeakersRange) {
        loadPodcastConfig();
    }

    // Voice cloning drag and drop
    if (dropZone && audioUpload) {
        dropZone.addEventListener('click', () => audioUpload.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                audioUpload.files = e.dataTransfer.files;
                handleAudioUpload(e.dataTransfer.files[0]);
            }
        });

        audioUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleAudioUpload(e.target.files[0]);
            }
        });
    }

    function handleAudioUpload(file) {
        if (!file) return;
        console.log('Audio file uploaded:', file.name);
        // Add your audio upload handling logic here
    }

    // Podcast Audio Handling
    function initPodcastAudio() {
        if (!podcastAudioContext) {
            podcastAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        }
        if (podcastAudioContext.state === 'suspended') {
            podcastAudioContext.resume();
        }

        // Setup visualizer if not already setup (or context recreated)
        if (!podcastVisualizer && document.getElementById('podcastVisualizer')) {
            const analyser = podcastAudioContext.createAnalyser();
            analyser.fftSize = 64;
            analyser.smoothingTimeConstant = 0.5;
            // We need to connect sources to this analyser, and analyser to destination
            // But playPodcastChunk connects source -> destination currently.
            // We need to change playPodcastChunk to connect source -> analyser -> destination

            // Attach analyser to context so we can reuse it
            podcastAudioContext.analyser = analyser;
            analyser.connect(podcastAudioContext.destination);

            podcastVisualizer = createVisualizer('podcastVisualizer', analyser);
            if (podcastVisualizer) podcastVisualizer.start();
        } else if (podcastVisualizer) {
            podcastVisualizer.start();
        }
    }

    async function playPodcastChunk(pcmData) {
        if (!podcastAudioContext) return;

        // Copy the data to ensure it persists
        const chunkCopy = new Int16Array(pcmData).slice();
        podcastChunks.push(chunkCopy);

        const pcm16 = new Int16Array(pcmData);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768.0;
        }

        const buffer = podcastAudioContext.createBuffer(1, float32.length, 24000);
        buffer.getChannelData(0).set(float32);

        const source = podcastAudioContext.createBufferSource();
        source.buffer = buffer;

        // Connect through analyser if available
        if (podcastAudioContext.analyser) {
            source.connect(podcastAudioContext.analyser);
        } else {
            source.connect(podcastAudioContext.destination);
        }

        const startTime = Math.max(podcastNextStartTime, podcastAudioContext.currentTime);
        source.start(startTime);
        podcastNextStartTime = startTime + buffer.duration;
    }

    function createPodcastWavBlob() {
        if (podcastChunks.length === 0) return null;

        const totalSamples = podcastChunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const dataLength = totalSamples * 2;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);

        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, 24000, true);
        view.setUint32(28, 24000 * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (const chunk of podcastChunks) {
            for (let i = 0; i < chunk.length; i++) {
                view.setInt16(offset, chunk[i], true);
                offset += 2;
            }
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    if (podcastGenerateBtn) {
        podcastGenerateBtn.addEventListener('click', () => {
            if (isPodcastGenerating) return;
            if (!podcastScript) {
                console.error('Podcast script element not found');
                return;
            }

            initPodcastAudio();
            isPodcastGenerating = true;
            podcastChunks = [];
            if (podcastLog) podcastLog.textContent = "🚀 Starting podcast generation...\n";
            if (podcastOutputCard) podcastOutputCard.style.opacity = "0.5";
            if (podcastDownloadBtn) podcastDownloadBtn.disabled = true;
            if (podcastPlayer) podcastPlayer.src = "";

            podcastGenerateBtn.disabled = true;
            if (podcastStopBtn) podcastStopBtn.disabled = false;
            podcastGenerateBtn.innerHTML = '<span class="btn-icon animate-spin">⏳</span> Generating...';

            // START METRICS
            const startTime = Date.now();
            const genTimeEl = document.getElementById('podcastGenTimeVal');
            const progressFill = document.getElementById('podcastProgressFill');
            const latencyEl = document.getElementById('podcastLatencyVal');
            const etaEl = document.getElementById('podcastEtaVal');
            let timer;

            if (genTimeEl) {
                // Reset
                genTimeEl.textContent = '0.0s';
                if (latencyEl) latencyEl.textContent = '-- ms';
                if (progressFill) progressFill.style.width = '0%';
                if (etaEl) etaEl.textContent = 'Calculating...';

                if (etaEl) etaEl.textContent = 'Estimating...';

                // Heuristic: ~3 seconds per line, ~0.6 chunks per char is too high for podcast.
                // Podcast chunks are large (probably). Let's guess based on chars.
                // 1 minute of audio is roughly 60s. TTS is fast, but streaming chunks come in.
                // Let's assume 15 chars = 1 second of audio.
                // Each chunk size depends on backend. VibeVoice chunks are usually 24000 samples (1s) or similar?
                // Let's rely on line count logic.
                const scriptLines = podcastScript.value.split('\n').filter(l => l.trim().length > 0).length;
                const estimatedDuration = scriptLines * 4; // 4 seconds per line average
                // If chunks are ~0.5s each, then total chunks = duration * 2
                const estimatedTotalChunks = Math.max(5, estimatedDuration * 4);
                let chunksReceived = 0;
                let firstChunkTime = null;

                timer = setInterval(() => {
                    const elapsed = (Date.now() - startTime) / 1000;
                    genTimeEl.textContent = elapsed.toFixed(1) + 's';
                    // Indeterminate or fake progress since we don't know total length initially
                    if (progressFill) {
                        // A slow creep effectively
                        // const p = Math.min(90, 100 * (1 - Math.exp(-elapsed / 30)));
                        // progressFill.style.width = p + '%';
                        // If we haven't received anything, show indeterminate progress
                        if (chunksReceived === 0) {
                            const p = Math.min(30, 100 * (1 - Math.exp(-elapsed / 10)));
                            progressFill.style.width = p + '%';
                        }
                    }
                }, 100);
            }

            if (podcastAudioContext) podcastNextStartTime = podcastAudioContext.currentTime + 0.1;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            podcastWs = new WebSocket(`${protocol}//${window.location.host}/podcast/ws`);
            podcastWs.binaryType = 'arraybuffer';

            // Store timer ref to clear later
            podcastWs.metricTimer = timer;

            podcastWs.onopen = () => {
                const speakers = [];
                const count = numSpeakersRange ? parseInt(numSpeakersRange.value) : 2;
                for (let i = 1; i <= count; i++) {
                    const el = document.getElementById(`speaker${i}`);
                    if (el) speakers.push(el.value);
                }

                podcastWs.send(JSON.stringify({
                    script: podcastScript ? podcastScript.value : "",
                    num_speakers: count,
                    speakers: speakers,
                    cfg: podcastCfgRange ? parseFloat(podcastCfgRange.value) : 1.5,
                    disable_cloning: disableCloning ? disableCloning.checked : false
                }));
            };

            podcastWs.onmessage = (event) => {
                // First chunk latency check
                if (podcastChunks.length === 0 && latencyEl) {
                    const lat = Date.now() - startTime;
                    latencyEl.textContent = lat + ' ms';
                    firstChunkTime = Date.now();
                }

                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'log') {
                        if (podcastLog) {
                            podcastLog.textContent += msg.message + "\n";
                            podcastLog.scrollTop = podcastLog.scrollHeight;
                        }
                    } else if (msg.type === 'complete' || msg.type === 'done') {
                        finishPodcast(true, startTime);
                    } else if (msg.type === 'error') {
                        if (podcastLog) podcastLog.textContent += "❌ " + msg.message + "\n";
                        finishPodcast(false);
                    }
                } else {
                    playPodcastChunk(event.data);
                    chunksReceived++;

                    // Update Progress & ETA
                    if (progressFill) {
                        const progress = Math.min(95, (chunksReceived / estimatedTotalChunks) * 100);
                        progressFill.style.width = Math.max(parseFloat(progressFill.style.width || 0), progress) + '%';
                    }

                    if (chunksReceived > 2 && firstChunkTime && etaEl) {
                        const elapsedSinceFirst = (Date.now() - firstChunkTime) / 1000;
                        // rate = chunks / sec
                        const rate = chunksReceived / elapsedSinceFirst;
                        if (rate > 0) {
                            const remaining = Math.max(0, estimatedTotalChunks - chunksReceived);
                            const timeLeft = remaining / rate;
                            etaEl.textContent = timeLeft.toFixed(1) + 's';
                        }
                    }
                }
            };

            podcastWs.onclose = () => { if (isPodcastGenerating) finishPodcast(false); };
            podcastWs.onerror = () => { if (isPodcastGenerating) finishPodcast(false); };
        });
    }

    if (podcastStopBtn) {
        podcastStopBtn.addEventListener('click', () => {
            if (podcastWs) {
                podcastWs.send(JSON.stringify({ type: 'stop' }));
                podcastWs.close();
            }
            finishPodcast(false);
        });
    }

    function finishPodcast(success, startTime) {
        isPodcastGenerating = false;
        if (podcastGenerateBtn) podcastGenerateBtn.disabled = false;
        if (podcastStopBtn) podcastStopBtn.disabled = true;
        if (podcastGenerateBtn) podcastGenerateBtn.innerHTML = '<span class="btn-icon">🎙️</span> Generate Podcast';

        // Stop Timer
        if (podcastWs && podcastWs.metricTimer) {
            clearInterval(podcastWs.metricTimer);
        }

        // Finalize metrics
        const genTimeEl = document.getElementById('podcastGenTimeVal');
        const progressFill = document.getElementById('podcastProgressFill');
        const etaEl = document.getElementById('podcastEtaVal');

        if (success) {
            if (startTime && genTimeEl) {
                const finalTime = (Date.now() - startTime) / 1000;
                genTimeEl.textContent = finalTime.toFixed(1) + 's';
            }
            if (progressFill) progressFill.style.width = '100%';
            if (etaEl) etaEl.textContent = 'Finished';

            if (podcastOutputCard) podcastOutputCard.style.opacity = "1";
            if (podcastDownloadBtn) podcastDownloadBtn.disabled = false;
            const blob = createPodcastWavBlob();
            if (blob && podcastPlayer) {
                podcastPlayer.src = URL.createObjectURL(blob);
            }
        } else {
            if (etaEl) etaEl.textContent = 'Stopped';
        }

        if (podcastWs) {
            podcastWs.onclose = null;
            podcastWs.onerror = null;
            podcastWs.close();
            podcastWs = null;
        }
    }

    if (podcastDownloadBtn) {
        podcastDownloadBtn.addEventListener('click', () => {
            const blob = createPodcastWavBlob();
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `vibevoice_podcast_${Date.now()}.wav`;
            link.click();
            URL.revokeObjectURL(url);
        });
    }

    /* -----------------------------------------------------------
       VOICE CLONING LOGIC
       ----------------------------------------------------------- */
    const cloneName = document.getElementById('cloneName');
    const cloneLang = document.getElementById('cloneLang');
    const cloneGender = document.getElementById('cloneGender');
    const cloneFile = document.getElementById('cloneFile');
    const cloneBtn = document.getElementById('cloneBtn');
    const cloneStatus = document.getElementById('cloneStatus');

    if (cloneBtn) {
        cloneBtn.addEventListener('click', async () => {
            if (!cloneName.value || !cloneFile.files[0]) {
                cloneStatus.textContent = "Please provide name and audio sample.";
                cloneStatus.style.color = "#ff6b6b";
                return;
            }

            const formData = new FormData();
            formData.append('file', cloneFile.files[0]);
            formData.append('name', cloneName.value);
            formData.append('language', cloneLang.value);
            formData.append('gender', cloneGender.value);

            cloneBtn.disabled = true;
            cloneBtn.innerHTML = "⏳ Cloning...";
            cloneStatus.textContent = "Uploading & processing...";
            cloneStatus.style.color = "var(--text-muted)";

            try {
                const res = await fetch('/clone_voice', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (res.ok) {
                    cloneStatus.textContent = "✅ " + data.message;
                    cloneStatus.style.color = "#4ade80";
                    cloneName.value = "";
                    cloneFile.value = "";

                    // Refresh voice lists
                    loadPodcastConfig();
                    loadClonedVoices();
                } else {
                    cloneStatus.textContent = "❌ Error: " + data.detail;
                    cloneStatus.style.color = "#ff6b6b";
                }
            } catch (e) {
                cloneStatus.textContent = "❌ Network Error";
                cloneStatus.style.color = "#ff6b6b";
            } finally {
                cloneBtn.disabled = false;
                cloneBtn.innerHTML = '<span class="btn-icon">🧬</span> Clone & Save Voice';
            }
        });
    }

    // Load cloned voices list
    async function loadClonedVoices() {
        const listDiv = document.getElementById('clonedVoicesList');
        if (!listDiv) return;

        try {
            const res = await fetch('/cloned_voices');
            const data = await res.json();

            if (data.voices && data.voices.length > 0) {
                listDiv.innerHTML = data.voices.map(voice => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; background: rgba(0,0,0,0.1);">
                        <div>
                            <strong>${voice.name}</strong> (${voice.language}, ${voice.gender})
                        </div>
                        <button class="delete-voice-btn" data-voice-id="${voice.id}" style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ff6b6b; color: #ff6b6b; padding: 5px 10px; border-radius: 4px; cursor: pointer;">
                            🗑️ Delete
                        </button>
                    </div>
                `).join('');

                // Add delete event listeners
                document.querySelectorAll('.delete-voice-btn').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const voiceId = e.target.dataset.voiceId;
                        if (confirm(`Delete voice "${voiceId}"?`)) {
                            try {
                                const res = await fetch(`/cloned_voices/${voiceId}`, { method: 'DELETE' });
                                if (res.ok) {
                                    loadClonedVoices();
                                    loadPodcastConfig();
                                } else {
                                    alert('Failed to delete voice');
                                }
                            } catch (e) {
                                alert('Network error');
                            }
                        }
                    });
                });
            } else {
                listDiv.innerHTML = '<div style="text-align: center; color: var(--text-muted);">No cloned voices yet.</div>';
            }
        } catch (e) {
            listDiv.innerHTML = '<div style="text-align: center; color: var(--text-muted);">Error loading voices.</div>';
        }
    }

    /* -----------------------------------------------------------
       TTS 1.5B LOGIC
       ----------------------------------------------------------- */
    const tts15bInput = document.getElementById('tts15bInput');
    const tts15bVoiceSelect = document.getElementById('tts15bVoiceSelect');
    const tts15bCfgRange = document.getElementById('tts15bCfgRange');
    const tts15bCfgVal = document.getElementById('tts15bCfgVal');
    const tts15bGenerateBtn = document.getElementById('tts15bGenerateBtn');
    const tts15bStatus = document.getElementById('tts15bStatus');
    const tts15bPlayer = document.getElementById('tts15bPlayer');
    const tts15bDownloadBtn = document.getElementById('tts15bDownloadBtn');
    const tts15bOutputCard = document.getElementById('tts15bOutputCard');

    if (tts15bCfgRange && tts15bCfgVal) {
        tts15bCfgRange.addEventListener('input', () => tts15bCfgVal.textContent = tts15bCfgRange.value);
    }


    // TTS 1.5B Audio & Visualizer
    // Variables already declared globally: tts15bAudioContext, tts15bAnalyser, tts15bVisualizer

    function initTts15bAudio() {
        if (!tts15bAudioContext) {
            tts15bAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        }
        if (tts15bAudioContext.state === 'suspended') {
            tts15bAudioContext.resume();
        }

        if (!tts15bVisualizer && document.getElementById('tts15bVisualizer')) {
            const analyser = tts15bAudioContext.createAnalyser();
            analyser.fftSize = 64;
            analyser.smoothingTimeConstant = 0.5;
            analyser.connect(tts15bAudioContext.destination);

            tts15bAnalyser = analyser;
            tts15bVisualizer = createVisualizer('tts15bVisualizer', analyser);

            // Connect player
            if (tts15bPlayer) {
                try {
                    const source = tts15bAudioContext.createMediaElementSource(tts15bPlayer);
                    source.connect(analyser);
                } catch (e) {
                    // Could fail if called multiple times or CORS issues (blob is local though)
                    console.log("TTS 1.5B source connection:", e);
                }

                tts15bPlayer.addEventListener('play', () => {
                    if (tts15bVisualizer) tts15bVisualizer.start();
                    if (tts15bAudioContext.state === 'suspended') tts15bAudioContext.resume();
                });
            }
        }
    }

    // Call init once to verify
    // But we should probably call it when tab is active or btn clicked

    if (document.getElementById('tts15bTab')) {
        document.getElementById('tts15bTab').addEventListener('click', initTts15bAudio);
    }

    // Sync voice list
    async function update15bVoices() {
        if (!tts15bVoiceSelect) return;
        // We reuse the podcast config logic to populate this
        try {
            const res = await fetch('/podcast/config');
            const data = await res.json();
            if (data.voices) {
                const current = tts15bVoiceSelect.value;
                tts15bVoiceSelect.innerHTML = data.voices.map(v =>
                    `<option value="${v}" ${v === current ? 'selected' : ''}>${v}</option>`
                ).join('');
            }
        } catch (e) { }
    }
    // Call it initially and whenever podcast config loads
    update15bVoices();
    // Hook into tab switch to refresh if needed
    document.querySelector('[data-tab="tts15b"]')?.addEventListener('click', update15bVoices);


    if (tts15bGenerateBtn) {
        tts15bGenerateBtn.addEventListener('click', async () => {
            if (!tts15bInput.value.trim()) {
                tts15bStatus.textContent = "Please enter some text.";
                return;
            }

            tts15bGenerateBtn.disabled = true;
            tts15bGenerateBtn.innerHTML = "⏳ Generating...";
            tts15bStatus.textContent = "Synthesis in progress (this may take a moment)...";
            if (tts15bOutputCard) tts15bOutputCard.style.opacity = "0.5";
            if (tts15bPlayer) tts15bPlayer.src = "";
            if (tts15bDownloadBtn) tts15bDownloadBtn.disabled = true;

            const formData = new FormData();
            formData.append('text', tts15bInput.value);
            formData.append('speaker', tts15bVoiceSelect.value);
            formData.append('cfg_scale', tts15bCfgRange.value);

            try {
                // START TIMER
                const startTime = Date.now();
                const genTimeEl = document.getElementById('tts15bGenTimeVal');
                const progressFill = document.getElementById('tts15bProgressFill');
                const etaEl = document.getElementById('tts15bEtaVal');
                let timer;

                // ETA Calculation (Heuristic)
                // Approx 0.1s per char + 2s base latency
                const estimatedTime = (tts15bInput.value.length * 0.1) + 2.0;

                if (genTimeEl) {
                    // Start fresh
                    genTimeEl.textContent = '0.0s';
                    if (etaEl) etaEl.textContent = estimatedTime.toFixed(1) + 's';
                    if (progressFill) progressFill.style.width = '0%';

                    timer = setInterval(() => {
                        const elapsed = (Date.now() - startTime) / 1000;
                        genTimeEl.textContent = elapsed.toFixed(1) + 's';

                        // Count down ETA
                        if (etaEl) {
                            const remaining = Math.max(0, estimatedTime - elapsed);
                            if (remaining < 0.5) etaEl.textContent = "Finishing...";
                            else etaEl.textContent = remaining.toFixed(1) + 's';
                        }

                        // Fake progress
                        if (progressFill) {
                            // S-curve approximation
                            const p = Math.min(95, 100 * (1 - Math.exp(-elapsed / (estimatedTime / 2))));
                            progressFill.style.width = p + '%';
                        }
                    }, 100);
                }

                // Initialize audio context if not yet
                initTts15bAudio();

                const res = await fetch('/tts_1_5b', {
                    method: 'POST',
                    body: formData
                });

                if (timer) clearInterval(timer);
                if (progressFill) progressFill.style.width = '100%';
                if (etaEl) etaEl.textContent = 'Finished';

                if (res.ok) {
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);

                    if (tts15bPlayer) tts15bPlayer.src = url;
                    if (tts15bOutputCard) tts15bOutputCard.style.opacity = "1";
                    if (tts15bDownloadBtn) {
                        tts15bDownloadBtn.disabled = false;
                        tts15bDownloadBtn.onclick = () => {
                            const link = document.createElement('a');
                            link.href = url;
                            link.download = `vibevoice_1.5b_${Date.now()}.wav`;
                            link.click();
                        };
                    }
                    tts15bStatus.textContent = "✅ Generation complete!";

                    // Show latency/final time
                    const totalTime = (Date.now() - startTime) / 1000;
                    if (genTimeEl) genTimeEl.textContent = totalTime.toFixed(1) + 's';

                    const latencyEl = document.getElementById('tts15bLatencyVal');
                    if (latencyEl) latencyEl.textContent = Math.round(totalTime * 1000) + ' ms';

                    if (tts15bPlayer) {
                        tts15bPlayer.play(); // Auto play which triggers visualizer
                    }

                } else {
                    const err = await res.text();
                    tts15bStatus.textContent = "❌ Error: " + err;
                }
            } catch (e) {
                tts15bStatus.textContent = "❌ Network Error";
                console.error(e);
            } finally {
                tts15bGenerateBtn.disabled = false;
                tts15bGenerateBtn.innerHTML = '<span class="btn-icon">✨</span> Generate (1.5B)';
            }
        });
    }

});