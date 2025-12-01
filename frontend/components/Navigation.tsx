'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 mb-8">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">📈</span>
            <span className="text-gray-900 dark:text-white font-bold text-xl">Stock Analysis</span>
          </div>
          <div className="flex space-x-4">
            <Link
              href="/"
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                pathname === '/'
                  ? 'bg-gray-900 text-white dark:bg-white dark:text-gray-900'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              Dashboard
            </Link>
            <Link
              href="/multi-stock"
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                pathname === '/multi-stock'
                  ? 'bg-gray-900 text-white dark:bg-white dark:text-gray-900'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              Multi-Stock AI
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
