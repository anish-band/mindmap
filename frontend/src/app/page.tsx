import Header from "@/components/Header";
import Hero from "@/components/Hero";
import WhatItDoes from "@/components/WhatItDoes";
import HowItWorks from "@/components/HowItWorks";
import LiveDemo from "@/components/LiveDemo";
import ModelComparison from "@/components/ModelComparison";
import Explainability from "@/components/Explainability";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="bg-gray-950 min-h-screen text-white overflow-x-hidden">
      <Header />
      <Hero />
      <WhatItDoes />
      <HowItWorks />
      <LiveDemo />
      <ModelComparison />
      <Explainability />
      <Footer />
    </main>
  );
}
