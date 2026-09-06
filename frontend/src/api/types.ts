export type TemplateType = 'comic' | 'map' | 'album'
export type VisualStyle = 'anime' | 'cyber' | 'realistic'

export interface UploadResponse { file_ids: string[] }
export interface StartTaskRequest {
  file_ids: string[]
  template: TemplateType
  style: VisualStyle
  description?: string
  materials?: string[]
}
export interface StartTaskResponse { task_id: string }
export interface TaskElement { id: string; url: string; label: string }
export type TaskStatusName = 'queued' | 'processing' | 'success' | 'failed'
export interface TaskStatus {
  task_id: string
  status: TaskStatusName
  progress: number
  elements: TaskElement[]
  result_url: string | null
  error?: string | null
}
export interface ComposeRequest { element_ids: string[] }
export interface ComposeResponse { result_url: string }
export interface Work {
  work_id: string
  cover_url: string
  template: TemplateType
  style: VisualStyle
  nickname: string
  likes: number
}
export interface WorkListResponse { items: Work[]; next_cursor?: string | null }
export interface PublishRequest { task_id: string; title: string }
export interface PublishResponse { work_id: string; share_url?: string }
