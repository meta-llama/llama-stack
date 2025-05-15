import { MessageSquareText, MessagesSquare, MoveUpRight } from "lucide-react";
import Link from "next/link";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";

const logItems = [
  {
    title: "Chat Completions",
    url: "/logs/chat-completions",
    icon: MessageSquareText,
  },
  {
    title: "Responses",
    url: "/logs/responses",
    icon: MessagesSquare,
  },
  {
    title: "Documentation",
    url: "https://llama-stack.readthedocs.io/en/latest/references/api_reference/index.html",
    icon: MoveUpRight,
  },
];

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>
        <Link href="/">Llama Stack</Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Logs</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {logItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
