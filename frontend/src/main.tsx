import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { CreationProvider } from './store'
import { EditorPage } from './pages/EditorPage'
import { GalleryPage } from './pages/GalleryPage'
import { PreviewPage } from './pages/PreviewPage'
import { TemplatePage } from './pages/TemplatePage'
import { UploadPage } from './pages/UploadPage'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><CreationProvider><BrowserRouter><Routes><Route element={<Layout/>}><Route path="/" element={<GalleryPage/>}/><Route path="/upload" element={<UploadPage/>}/><Route path="/template" element={<TemplatePage/>}/><Route path="/editor" element={<EditorPage/>}/><Route path="/preview" element={<PreviewPage/>}/></Route></Routes></BrowserRouter></CreationProvider></React.StrictMode>)
