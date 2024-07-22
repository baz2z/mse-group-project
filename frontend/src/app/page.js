import Image from "next/image";

import SearchComponent from "@/components/SearchComponent";
import './page-module.css';

import { Lustria } from 'next/font/google';
const lustria = Lustria({
  weight: '400', // specify weights if needed
  subsets: ['latin'], // specify subsets if needed
});

export default function Home() {
  return (
    <>
    {/* <main style={{ backgroundImage: `url(/tuebingen_neckar_1.jpg)`, backgroundSize: 'cover', filter: 'blur(1px)' }} className="custom-container flex min-h-screen flex-col items-center p-24">
    </main> */}
      <div className="custom-container flex min-h-screen flex-col items-center p-24 pt-32">
      <header className="text-center">
      <h1 className="text-4xl font-bold text-white tracking-tight" style={{ textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)' }}>
        A gateway to the Heart of Tübingen
      </h1>
      <h2 className="text-lg mt-8 text-white" style={{ fontFamily: lustria.style.fontFamily, textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)' }}>
        search to get relevant information on Tübingen
      </h2>
    </header>
      <div className="mt-14">
          <SearchComponent />
        </div>
      </div>
      </>
  );
}
