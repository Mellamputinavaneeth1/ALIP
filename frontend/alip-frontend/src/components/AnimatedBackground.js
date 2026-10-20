import React, { useRef, useEffect } from 'react';

/**
 * AnimatedBackground — Slow Floating Leaves
 * Minimalist animation featuring sparse, slow-moving wind-blown leaves drifting across a deep background.
 */
export default function AnimatedBackground({ diseaseDetected = false }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const diseasedRef = useRef(diseaseDetected);

  useEffect(() => {
    diseasedRef.current = diseaseDetected;
  }, [diseaseDetected]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W = (canvas.width = window.innerWidth);
    let H = (canvas.height = window.innerHeight);

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    // ── Floating Leaves System ──
    const spawnLeaf = (randomProgression = false) => {
      const birthX = randomProgression ? (Math.random() * W) : (-40 - Math.random() * 80);
      const birthY = Math.random() * H;

      return {
        x: birthX,
        y: birthY,
        size: 9 + Math.random() * 12,
        vx: 0.5 + Math.random() * 0.7, // Reduced velocity
        vy: 0.1 + Math.random() * 0.3, // Reduced vertical velocity
        angle: Math.random() * Math.PI * 2,
        vAngle: (Math.random() - 0.5) * 0.02, // Reduced rotation speed
        flip: Math.random() * Math.PI,
        vFlip: 0.015 + Math.random() * 0.02, // Reduced flip speed
        swayPhase: Math.random() * Math.PI * 2,
        colorType: Math.random(),
      };
    };

    const LEAF_COUNT = 15; // Reduced quantity
    const LEAVES = Array.from({ length: LEAF_COUNT }, () => spawnLeaf(true));

    let visible = true;
    const onVis = () => { visible = !document.hidden; };
    document.addEventListener('visibilitychange', onVis);

    let time = 0;

    const render = () => {
      rafRef.current = requestAnimationFrame(render);
      if (!visible) return;
      time += 0.016;

      const isDiseased = diseasedRef.current;
      const baseWind = isDiseased ? 1.0 : 0.6; // Reduced windspeed

      ctx.clearRect(0, 0, W, H);

      // Deep Sky Gradient
      const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
      if (!isDiseased) {
        bgGrad.addColorStop(0, '#060f1b');
        bgGrad.addColorStop(0.5, '#0b192c');
        bgGrad.addColorStop(0.85, '#0e242d');
        bgGrad.addColorStop(1, '#091c1b');
      } else {
        bgGrad.addColorStop(0, '#130d1c');
        bgGrad.addColorStop(0.5, '#1e1424');
        bgGrad.addColorStop(0.85, '#1b1c1d');
        bgGrad.addColorStop(1, '#111816');
      }
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // Draw Floating Leaves
      LEAVES.forEach(p => {
        p.x += p.vx * baseWind;
        p.y += p.vy + Math.sin(time * 1.5 + p.swayPhase) * 0.3; // Slower sway
        p.angle += p.vAngle * baseWind;
        p.flip += p.vFlip * baseWind;

        if (p.x > W + 40 || p.y > H + 40) {
          Object.assign(p, spawnLeaf(false));
        }

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.angle);
        ctx.scale(1, Math.cos(p.flip));

        ctx.beginPath();
        
        // When diseased, shift overall colors toward autumn/amber tones
        if (!isDiseased && p.colorType > 0.85) {
          // Blossom petal
          ctx.moveTo(0, -p.size);
          ctx.quadraticCurveTo(p.size * 0.8, -p.size * 0.3, 0, p.size * 0.8);
          ctx.quadraticCurveTo(-p.size * 0.8, -p.size * 0.3, 0, -p.size);
          ctx.closePath();
          ctx.fillStyle = 'rgba(254, 215, 226, 0.85)'; // Soft pink/white
          ctx.fill();
        } else {
          // Leaf shape
          ctx.moveTo(0, -p.size);
          ctx.quadraticCurveTo(p.size * 0.65, 0, 0, p.size);
          ctx.quadraticCurveTo(-p.size * 0.65, 0, 0, -p.size);
          ctx.closePath();
          
          if (isDiseased) {
            ctx.fillStyle = p.colorType > 0.5 ? 'rgba(234, 179, 8, 0.75)' : 'rgba(217, 119, 6, 0.75)';
          } else if (p.colorType > 0.7) {
            ctx.fillStyle = 'rgba(251, 191, 36, 0.78)'; // Golden
          } else {
            ctx.fillStyle = 'rgba(52, 211, 153, 0.75)'; // Fresh green
          }
          ctx.fill();

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.38)';
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(0, -p.size * 0.8);
          ctx.lineTo(0, p.size * 0.8);
          ctx.stroke();
        }

        ctx.restore();
      });
    };

    rafRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
      aria-hidden="true"
    />
  );
}
