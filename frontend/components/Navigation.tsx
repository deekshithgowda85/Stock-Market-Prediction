'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-gray-800 border-b border-gray-700 mb-8">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">📈</span>
            <span className="text-white font-bold text-xl">Stock Analysis</span>
          </div>
          <div className="flex space-x-4">
            <Link
              href="/"
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                pathname === '/'
                  ? 'bg-white text-gray-900'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              Dashboard
            </Link>
            <Link
              href="/pipeline"
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                pathname === '/pipeline'
                  ? 'bg-white text-gray-900'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              Pipeline Control
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
