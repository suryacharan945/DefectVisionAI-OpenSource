import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Sparkles } from '@react-three/drei';
import * as THREE from 'three';

function SensorCore({ mousePos }) {
  const outerRingRef = useRef();
  const midRingRef = useRef();
  const innerCoreRef = useRef();
  const lightRef = useRef();

  useFrame((state, delta) => {
    // Gentle rotation
    if (outerRingRef.current) {
      outerRingRef.current.rotation.x += delta * 0.25;
      outerRingRef.current.rotation.y += delta * 0.15;
    }
    if (midRingRef.current) {
      midRingRef.current.rotation.y -= delta * 0.35;
      midRingRef.current.rotation.z += delta * 0.2;
    }
    if (innerCoreRef.current) {
      innerCoreRef.current.rotation.x += delta * 0.5;
      innerCoreRef.current.rotation.z += delta * 0.4;
    }

    // Parallax damping to mouse coordinates
    const targetX = (mousePos.x * Math.PI) / 12;
    const targetY = (mousePos.y * Math.PI) / 12;
    state.camera.position.x = THREE.MathUtils.lerp(state.camera.position.x, targetX * 1.5, 0.05);
    state.camera.position.y = THREE.MathUtils.lerp(state.camera.position.y, -targetY * 1.5, 0.05);
    state.camera.lookAt(0, 0, 0);
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Outer Gimbal Ring (Cyan Laser Ring) */}
      <mesh ref={outerRingRef}>
        <torusGeometry args={[2.5, 0.035, 24, 100]} />
        <meshStandardMaterial
          color="#06b6d4"
          emissive="#0891b2"
          emissiveIntensity={1.2}
          roughness={0.2}
          metalness={0.9}
        />
      </mesh>

      {/* Middle Gyro Ring (Electric Violet Ring) */}
      <mesh ref={midRingRef}>
        <torusGeometry args={[1.9, 0.045, 24, 100]} />
        <meshStandardMaterial
          color="#8b5cf6"
          emissive="#7c3aed"
          emissiveIntensity={1.5}
          roughness={0.2}
          metalness={0.95}
        />
      </mesh>

      {/* Core Octahedron / Crystal Anomaly Sensor */}
      <Float speed={2} rotationIntensity={1} floatIntensity={1.5}>
        <mesh ref={innerCoreRef} scale={0.9}>
          <octahedronGeometry args={[1.1, 0]} />
          <meshStandardMaterial
            color="#22d3ee"
            emissive="#a855f7"
            emissiveIntensity={0.8}
            roughness={0.1}
            metalness={0.9}
            wireframe={true}
          />
        </mesh>

        {/* Central Glowing Energy Nucleus */}
        <mesh scale={0.65}>
          <sphereGeometry args={[0.7, 32, 32]} />
          <meshStandardMaterial
            color="#c084fc"
            emissive="#8b5cf6"
            emissiveIntensity={2.0}
            roughness={0.3}
            metalness={0.8}
          />
        </mesh>
      </Float>

      {/* Floating Inspection Data Particles */}
      <Sparkles count={80} scale={6} size={2.5} speed={0.4} color="#38bdf8" />
      <Sparkles count={50} scale={4} size={3} speed={0.6} color="#a855f7" />

      {/* Internal Point Light */}
      <pointLight ref={lightRef} position={[0, 0, 0]} color="#8b5cf6" intensity={4} distance={8} />
    </group>
  );
}

export default function IndustrialCore3D() {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [hasWebGL, setHasWebGL] = useState(true);

  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = (e.clientY / window.innerHeight) * 2 - 1;
      setMousePos({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  if (!hasWebGL) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="w-64 h-64 rounded-full bg-gradient-to-tr from-violet-600 via-cyan-500 to-indigo-700 blur-3xl opacity-40 animate-pulse" />
      </div>
    );
  }

  return (
    <div className="w-full h-full relative cursor-grab active:cursor-grabbing">
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        onError={() => setHasWebGL(false)}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1.2} color="#38bdf8" />
        <directionalLight position={[-10, -10, -5]} intensity={0.8} color="#8b5cf6" />
        <SensorCore mousePos={mousePos} />
      </Canvas>
    </div>
  );
}
