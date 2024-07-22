import "./globals.css";
import Head from 'next/head';
import { library } from '@fortawesome/fontawesome-svg-core';
import { fab } from '@fortawesome/free-brands-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

library.add(fab);

export const metadata = {
  title: "Tübingen Search Engine",
  description: "Modern Search Engine for Tübingen",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <Head>
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
          integrity="sha384-k6RqeWeci5ZR/Lv4MR0sA0FfDOMo8O4nBmg6JZRf2x4OjmHtTEjM8iGl3TE7Bg5M"
          crossOrigin="anonymous"
        />
        <link rel="icon" type="image/png" sizes="16x16" href="/images/favicon.png"/>
      </Head>
      <body className="flex flex-col min-h-screen">
        <nav className="top-0 left-0 w-full z-20 bg-gradient-to-r from-white/20 to-white/10 backdrop-blur-md shadow-lg sticky">
          <div className="container mx-auto flex items-center justify-between p-4 pl-10 pr-1">
            <div className="flex items-center">
              <img src="/uni_tuebingen_logo_black_white.png" alt="Tübingen Search Engine Logo" className="h-12 w-auto" />
            </div>
            <a href="https://github.com/baz2z/mse-group-project" target="_blank" rel="noopener noreferrer" className="text-gray-800 hover:text-gray-600">
              <FontAwesomeIcon icon={['fab', 'github']} size="2x" className="w-8 h-8" />
            </a>
          </div>
        </nav>

        <main style={{ backgroundImage: `url(/tuebingen_neckar_1.jpg)`, backgroundSize: 'cover', filter: 'blur(3px)' }} 
              className="fixed left-0 right-0 flex min-h-screen flex-col items-center p-24">
        </main>
        <main className="fixed left-0 right-0 flex min-h-screen flex-col items-center p-24">
          <div className="absolute inset-0 bg-cover" style={{ backgroundImage: `url(/tuebingen_neckar_1.jpg)`, filter: 'blur(3px)' }}></div>
        </main>
        {children}
      </body>
    </html>
  );
}
