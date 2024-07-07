import Image from "next/image";
import styles from './page-module.css';
import SearchComponent from "@/components/SearchComponent";

import { Lustria } from 'next/font/google';
const lustria = Lustria({
  weight: '400', // specify weights if needed
  subsets: ['latin'], // specify subsets if needed
});

export default function Home() {
  return (
    <main style={{ backgroundImage: `url(/tuebingen_neckar_1.jpg)`, backgroundSize: 'cover' }} className="flex min-h-screen flex-col items-center p-24">
      <header className="text-center">
        <h1 className="text-4xl font-bold">Tübingen</h1>
        <h2 className="text-2xl mt-2" style={{ fontFamily: lustria.style.fontFamily }}>Where wisdom meets charm</h2>
      </header>
      <div className="mt-8">
          <SearchComponent />
        </div>

    </main>
  );
}
