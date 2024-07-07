import "./globals.css";
import Head from 'next/head';

// import { Inter } from "next/font/google";
// const inter = Inter({ subsets: ["latin"] });

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
          crossorigin="anonymous"
        />
      </Head>
      {/* className={inter.className} */}
      <body>{children}</body>
    </html>
  );
}
