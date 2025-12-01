"use client" 

import * as React from "react"
import { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { Menu, X, TrendingUp } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import ThemeToggle from "@/components/ThemeToggle"

const Navbar1 = () => {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()

  const toggleMenu = () => setIsOpen(!isOpen)
  
  const navItems = [
    { name: "Dashboard", href: "/" },
    { name: "Multi-Stock AI", href: "/multi-stock" },
  ]

  return (
    <div className="sticky top-0 z-50 flex justify-center py-6 px-4">
      <div className="flex items-center justify-between px-6 py-3 bg-[hsl(var(--card))] rounded-full shadow-lg w-full max-w-6xl relative border border-[hsl(var(--border))]">
        {/* Logo */}
        <Link href="/" className="flex items-center">
          <motion.div
            className="w-8 h-8 mr-3 flex items-center justify-center"
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            whileHover={{ rotate: 10 }}
            transition={{ duration: 0.3 }}
          >
            <TrendingUp className="h-6 w-6 text-blue-500" />
          </motion.div>
          <span className="text-lg font-bold text-[hsl(var(--card-foreground))]">
            Stock AI
          </span>
        </Link>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-8">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            return (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                whileHover={{ scale: 1.05 }}
              >
                <Link 
                  href={item.href} 
                  className={`text-sm transition-colors font-medium ${
                    isActive 
                      ? 'text-blue-600' 
                      : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                  }`}
                >
                  {item.name}
                </Link>
              </motion.div>
            )
          })}
        </nav>

        {/* Desktop Status Badge + Theme Toggle */}
        <motion.div
          className="hidden md:flex items-center gap-3"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <div className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 rounded-full border border-green-300 dark:border-green-700">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-green-700 dark:text-green-400">
              Live
            </span>
          </div>
          <ThemeToggle />
        </motion.div>

        {/* Mobile Menu Button */}
        <motion.button 
          className="md:hidden flex items-center" 
          onClick={toggleMenu} 
          whileTap={{ scale: 0.9 }}
        >
          <Menu className="h-6 w-6 text-[hsl(var(--foreground))]" />
        </motion.button>
      </div>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 bg-[hsl(var(--background))] z-50 pt-24 px-6 md:hidden"
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
          >
            <motion.button
              className="absolute top-6 right-6 p-2"
              onClick={toggleMenu}
              whileTap={{ scale: 0.9 }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <X className="h-6 w-6 text-[hsl(var(--foreground))]" />
            </motion.button>
              <div className="flex flex-col space-y-6">
              {navItems.map((item, i) => {
                const isActive = pathname === item.href
                return (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 + 0.1 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    <Link 
                      href={item.href} 
                      className={`text-base font-medium ${
                        isActive 
                          ? 'text-blue-600' 
                          : 'text-[hsl(var(--foreground))]'
                      }`}
                      onClick={toggleMenu}
                    >
                      {item.name}
                    </Link>
                  </motion.div>
                )
              })}

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                exit={{ opacity: 0, y: 20 }}
                className="pt-6"
              >
                <div className="flex items-center gap-2 px-5 py-3 bg-green-100 dark:bg-green-900/30 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-sm font-medium text-green-700 dark:text-green-400">
                    System Live
                  </span>
                </div>
              </motion.div>

              {/* Theme Toggle (mobile) */}
              <div className="pt-4">
                <ThemeToggle />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


export { Navbar1 }
