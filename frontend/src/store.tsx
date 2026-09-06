import { createContext, useContext, useMemo, useState, type PropsWithChildren } from 'react'
import type { TaskElement, TemplateType, VisualStyle } from './api'

interface CreationState {
  files: File[]; description: string; fileIds: string[]; template: TemplateType; style: VisualStyle;
  materials: string[]; taskId: string; elements: TaskElement[]; selectedIds: string[]; resultUrl: string; title: string
}
interface CreationContextValue extends CreationState { update: (patch: Partial<CreationState>) => void }
const initial: CreationState = { files: [], description: '', fileIds: [], template: 'comic', style: 'anime', materials: ['badge'], taskId: '', elements: [], selectedIds: [], resultUrl: '', title: '我的活动纪念' }
const CreationContext = createContext<CreationContextValue | null>(null)

export function CreationProvider({ children }: PropsWithChildren) {
  const [state, setState] = useState(initial)
  const value = useMemo(() => ({ ...state, update: (patch: Partial<CreationState>) => setState((old) => ({ ...old, ...patch })) }), [state])
  return <CreationContext.Provider value={value}>{children}</CreationContext.Provider>
}
export function useCreation() { const value = useContext(CreationContext); if (!value) throw new Error('CreationProvider missing'); return value }
