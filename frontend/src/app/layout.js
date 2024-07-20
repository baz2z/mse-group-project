import "./globals.css";
import Head from 'next/head';

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
      </Head>
      <body className="flex flex-col min-h-screen">
        <nav className="absolute top-0 left-0 w-full bg-transparent z-10">
          <div className="container mx-auto flex items-center p-4 pl-14">
            <div className="flex items-center">
              <img src="/uni_tuebingen_logo_black_white.png" alt="Tübingen Search Engine Logo" className="h-16 w-auto" />
            </div>
          </div>
        </nav>
        <main className="flex flex-1 flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
